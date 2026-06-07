"""Council mode — multi-model deliberation before task execution.

``context_mode: council`` runs N participants through R rounds of
discussion, then consolidates their perspectives into a single context
document for the downstream task.

Topologies:
- **star**: all participants see each other's responses after every round.
- **chain**: participants execute sequentially; each sees only the prior
  participant's response.  No consolidation — the last output IS the
  synthesis.
- **graph**: like star but each participant only sees responses from peers
  listed in the ``connections`` map.  Consolidation runs at the end.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .models import (
    EngineDefaults,
    EngineName,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

CouncilTopology = Literal["star", "chain", "graph"]

_VALID_TOPOLOGIES: set[str] = {"star", "chain", "graph"}
_MAX_ROUNDS = 5
_COUNCIL_TIMEOUT = 120  # seconds per participant call
_CONSOLIDATION_MODEL = "haiku"


@dataclass
class CouncilParticipant:
    """One member of a council panel."""

    engine: str  # EngineName
    model: str | None = None
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "model": self.model, "role": self.role}


@dataclass
class CouncilSpec:
    """Configuration for a council deliberation."""

    participants: list[CouncilParticipant] = field(default_factory=list)
    rounds: int = 1
    topology: str = "star"
    consensus_threshold: float = 0.7
    connections: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "participants": [p.to_dict() for p in self.participants],
            "rounds": self.rounds,
            "topology": self.topology,
            "consensus_threshold": self.consensus_threshold,
        }
        if self.connections:
            d["connections"] = self.connections
        return d


@dataclass
class CouncilRound:
    """Responses from one round of deliberation."""

    round_num: int
    responses: dict[str, str] = field(default_factory=dict)  # role → response


@dataclass
class CouncilResult:
    """Full council deliberation result."""

    rounds: list[CouncilRound] = field(default_factory=list)
    synthesis: str = ""
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": [
                {"round_num": r.round_num, "responses": r.responses}
                for r in self.rounds
            ],
            "synthesis": self.synthesis,
            "total_cost_usd": self.total_cost_usd,
        }


# ---------------------------------------------------------------------------
# Engine call
# ---------------------------------------------------------------------------


def _call_participant(
    participant: CouncilParticipant,
    prompt: str,
    workdir: Path,
    timeout_sec: int = _COUNCIL_TIMEOUT,
) -> tuple[str, float | None]:
    """Execute a single participant call via engine CLI.

    Returns ``(response_text, cost_or_none)``.  Never raises — returns
    an error message on failure.
    """
    from .runners import _build_safe_env, _extract_cost_from_line, _resolve_executable, build_command

    # Build a minimal plan + task stub for build_command()
    defaults = PlanDefaults()
    engine_defaults = EngineDefaults(model=participant.model)
    setattr(defaults, participant.engine, engine_defaults)
    plan = PlanSpec(name="council", defaults=defaults)

    task = TaskSpec(
        id=f"council-{participant.role or participant.engine}",
        engine=participant.engine,  # type: ignore[arg-type]
        model=participant.model,
        prompt=prompt,
    )

    try:
        command, _ = build_command(plan, task, workdir, execution_profile="safe")
    except Exception as exc:
        return f"[council] error building command for {participant.role}: {exc}", None

    # Adjust command for non-interactive output
    if isinstance(command, list):
        command = _adjust_for_council(command, participant.engine)

    env = _build_safe_env({}, {})

    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
            shell=isinstance(command, str),
        )
    except subprocess.TimeoutExpired:
        return f"[council] {participant.role} timed out after {timeout_sec}s", None
    except FileNotFoundError:
        return f"[council] engine CLI not found for '{participant.engine}'", None
    except Exception as exc:
        return f"[council] error calling {participant.role}: {exc}", None

    output = proc.stdout.strip() if proc.stdout else ""
    if not output and proc.stderr:
        output = proc.stderr.strip()

    # Extract result text (handle stream-json for claude)
    if participant.engine == "claude" and output:
        from .runners import _extract_stream_json_result_text
        extracted = _extract_stream_json_result_text(output)
        if extracted:
            output = extracted

    # Extract cost
    cost: float | None = None
    for line in reversed((output or "").splitlines()[-20:]):
        c = _extract_cost_from_line(line)
        if c is not None:
            cost = c
            break

    return output, cost


def _adjust_for_council(cmd: list[str], engine: str) -> list[str]:
    """Adjust engine command for non-interactive council output."""
    result: list[str] = []
    skip_next = False
    for i, arg in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        # Replace stream-json with text for claude
        if arg == "--output-format" and i + 1 < len(cmd) and cmd[i + 1] in ("json", "stream-json"):
            result.append("--output-format")
            result.append("text")
            skip_next = True
            continue
        result.append(arg)
    return result


# ---------------------------------------------------------------------------
# Round prompts
# ---------------------------------------------------------------------------


def _build_round_prompt(
    base_prompt: str,
    participant: CouncilParticipant,
    round_num: int,
    prior_rounds: list[CouncilRound],
    upstream_context: str,
) -> str:
    """Build the prompt for a participant in a specific round."""
    parts: list[str] = []

    # Role context (specific focus area, not a generic persona)
    if participant.role:
        parts.append(f"Your focus area in this deliberation is **{participant.role}**. Prioritise that lens in your analysis.")

    parts.append(f"Round {round_num}. Provide your perspective on the following task.\n")

    # Upstream context (if any)
    if upstream_context:
        parts.append(f"<upstream_context>\n{upstream_context}\n</upstream_context>\n")

    # Prior round responses
    if prior_rounds:
        discussion: list[str] = []
        for pr in prior_rounds:
            for role, resp in pr.responses.items():
                # Truncate very long responses
                truncated = resp[:4000] if len(resp) > 4000 else resp
                discussion.append(f"**{role}** (round {pr.round_num}): {truncated}")
        parts.append(
            "<prior_discussion>\n"
            + "\n\n".join(discussion)
            + "\n</prior_discussion>\n"
        )
        parts.append(
            "Consider the perspectives above. Build on points of agreement, "
            "address disagreements, and refine your position.\n"
        )

    # The actual task
    parts.append(f"<task>\n{base_prompt}\n</task>")

    return "\n".join(parts)


def _build_consolidation_prompt(
    base_prompt: str,
    all_rounds: list[CouncilRound],
) -> str:
    """Build the final consolidation prompt."""
    discussion: list[str] = []
    for r in all_rounds:
        for role, resp in r.responses.items():
            truncated = resp[:4000] if len(resp) > 4000 else resp
            discussion.append(f"**{role}** (round {r.round_num}): {truncated}")

    return (
        "You are the consolidator for a council deliberation. "
        "Synthesize the following perspectives into a unified, actionable response.\n\n"
        f"<original_task>\n{base_prompt}\n</original_task>\n\n"
        f"<council_discussion>\n"
        + "\n\n".join(discussion)
        + "\n</council_discussion>\n\n"
        "Provide a single consolidated response that:\n"
        "1. Captures the consensus where participants agreed\n"
        "2. Notes important dissenting views\n"
        "3. Recommends a clear course of action\n"
    )


# ---------------------------------------------------------------------------
# Chain topology prompt
# ---------------------------------------------------------------------------


def _build_chain_prompt(
    base_prompt: str,
    participant: CouncilParticipant,
    prior_response: str | None,
    upstream_context: str,
) -> str:
    """Build the prompt for a participant in a chain topology.

    Each participant sees only the previous participant's response.
    """
    parts: list[str] = []

    if participant.role:
        parts.append(f"Your focus area in this chain is **{participant.role}**. Prioritise that lens when refining the response.")

    if upstream_context:
        parts.append(f"<upstream_context>\n{upstream_context}\n</upstream_context>\n")

    if prior_response is not None:
        truncated = prior_response[:4000] if len(prior_response) > 4000 else prior_response
        parts.append(
            f"<prior_response>\n"
            f"Previous participant responded:\n{truncated}\n"
            f"</prior_response>\n"
            f"Refine and improve this response.\n"
        )

    parts.append(f"<task>\n{base_prompt}\n</task>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Graph topology prompt
# ---------------------------------------------------------------------------


def _build_graph_round_prompt(
    base_prompt: str,
    participant: CouncilParticipant,
    round_num: int,
    prior_rounds: list[CouncilRound],
    connections: dict[str, list[str]],
    upstream_context: str,
) -> str:
    """Build the prompt for a participant in a graph topology.

    Like ``_build_round_prompt`` but filters prior responses to only those
    from connected peers as defined by the *connections* map.
    """
    role_key = participant.role or f"{participant.engine}-{participant.model or 'default'}"
    visible_peers = set(connections.get(role_key, []))

    parts: list[str] = []

    if participant.role:
        parts.append(f"Your focus area in this deliberation is **{participant.role}**. Prioritise that lens in your analysis.")

    parts.append(f"Round {round_num}. Provide your perspective on the following task.\n")

    if upstream_context:
        parts.append(f"<upstream_context>\n{upstream_context}\n</upstream_context>\n")

    if prior_rounds and visible_peers:
        discussion: list[str] = []
        for pr in prior_rounds:
            for role, resp in pr.responses.items():
                if role not in visible_peers:
                    continue
                truncated = resp[:4000] if len(resp) > 4000 else resp
                discussion.append(f"**{role}** (round {pr.round_num}): {truncated}")
        if discussion:
            parts.append(
                "<prior_discussion>\n"
                + "\n\n".join(discussion)
                + "\n</prior_discussion>\n"
            )
            parts.append(
                "Consider the perspectives above. Build on points of agreement, "
                "address disagreements, and refine your position.\n"
            )

    parts.append(f"<task>\n{base_prompt}\n</task>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Topology runners
# ---------------------------------------------------------------------------


def _run_star_council(
    spec: CouncilSpec,
    base_prompt: str,
    workdir: Path,
    upstream_context: str,
    event_callback: Any,
) -> CouncilResult:
    """Star topology: all participants see all responses each round."""
    result = CouncilResult()

    for round_num in range(1, spec.rounds + 1):
        council_round = CouncilRound(round_num=round_num)

        for participant in spec.participants:
            prompt = _build_round_prompt(
                base_prompt,
                participant,
                round_num,
                result.rounds,
                upstream_context,
            )

            if event_callback:
                event_callback(
                    "council_turn",
                    round=round_num,
                    total_rounds=spec.rounds,
                    participant_role=participant.role,
                    participant_engine=participant.engine,
                )

            response, cost = _call_participant(participant, prompt, workdir)
            role_key = participant.role or f"{participant.engine}-{participant.model or 'default'}"
            council_round.responses[role_key] = response

            if cost is not None:
                result.total_cost_usd += cost

        result.rounds.append(council_round)

    # Consolidation
    if result.rounds:
        consolidation_prompt = _build_consolidation_prompt(base_prompt, result.rounds)
        consolidator = CouncilParticipant(engine="claude", model=_CONSOLIDATION_MODEL, role="consolidator")

        if event_callback:
            event_callback(
                "council_consolidation",
                total_rounds=spec.rounds,
                participant_count=len(spec.participants),
            )

        synthesis, cost = _call_participant(consolidator, consolidation_prompt, workdir)
        result.synthesis = synthesis
        if cost is not None:
            result.total_cost_usd += cost

    return result


def _run_chain_council(
    spec: CouncilSpec,
    base_prompt: str,
    workdir: Path,
    upstream_context: str,
    event_callback: Any,
) -> CouncilResult:
    """Chain topology: sequential pass — each sees only the prior response.

    No consolidation step; the last participant's output is the synthesis.
    """
    result = CouncilResult()
    prior_response: str | None = None
    council_round = CouncilRound(round_num=1)

    for step_idx, participant in enumerate(spec.participants):
        prompt = _build_chain_prompt(
            base_prompt, participant, prior_response, upstream_context,
        )

        if event_callback:
            event_callback(
                "council_chain_step",
                task_id="",
                participant_role=participant.role,
                step_index=step_idx,
            )

        response, cost = _call_participant(participant, prompt, workdir)
        role_key = participant.role or f"{participant.engine}-{participant.model or 'default'}"
        council_round.responses[role_key] = response
        prior_response = response

        if cost is not None:
            result.total_cost_usd += cost

    result.rounds.append(council_round)
    # Last participant's output is the synthesis — no consolidation
    result.synthesis = prior_response or ""
    return result


def _run_graph_council(
    spec: CouncilSpec,
    base_prompt: str,
    workdir: Path,
    upstream_context: str,
    event_callback: Any,
) -> CouncilResult:
    """Graph topology: like star but visibility is constrained by connections."""
    result = CouncilResult()

    for round_num in range(1, spec.rounds + 1):
        council_round = CouncilRound(round_num=round_num)

        for participant in spec.participants:
            prompt = _build_graph_round_prompt(
                base_prompt,
                participant,
                round_num,
                result.rounds,
                spec.connections,
                upstream_context,
            )

            if event_callback:
                event_callback(
                    "council_turn",
                    round=round_num,
                    total_rounds=spec.rounds,
                    participant_role=participant.role,
                    participant_engine=participant.engine,
                )

            response, cost = _call_participant(participant, prompt, workdir)
            role_key = participant.role or f"{participant.engine}-{participant.model or 'default'}"
            council_round.responses[role_key] = response

            if cost is not None:
                result.total_cost_usd += cost

        result.rounds.append(council_round)

    # Consolidation (same as star)
    if result.rounds:
        consolidation_prompt = _build_consolidation_prompt(base_prompt, result.rounds)
        consolidator = CouncilParticipant(engine="claude", model=_CONSOLIDATION_MODEL, role="consolidator")

        if event_callback:
            event_callback(
                "council_consolidation",
                total_rounds=spec.rounds,
                participant_count=len(spec.participants),
            )

        synthesis, cost = _call_participant(consolidator, consolidation_prompt, workdir)
        result.synthesis = synthesis
        if cost is not None:
            result.total_cost_usd += cost

    return result


# ---------------------------------------------------------------------------
# Main council execution
# ---------------------------------------------------------------------------


def run_council(
    spec: CouncilSpec,
    base_prompt: str,
    workdir: Path,
    upstream_context: str = "",
    event_callback: Any = None,
) -> CouncilResult:
    """Execute a council deliberation, dispatching by topology.

    - ``star``: all participants see all responses each round, then consolidate.
    - ``chain``: sequential pipeline — each sees only the prior response; no
      consolidation.
    - ``graph``: like star but visibility constrained by ``connections``; then
      consolidate.

    Returns a ``CouncilResult`` with the full discussion and synthesis.
    """
    if spec.topology == "chain":
        return _run_chain_council(spec, base_prompt, workdir, upstream_context, event_callback)
    if spec.topology == "graph":
        return _run_graph_council(spec, base_prompt, workdir, upstream_context, event_callback)
    # Default: star
    return _run_star_council(spec, base_prompt, workdir, upstream_context, event_callback)
