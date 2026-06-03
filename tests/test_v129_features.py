"""Tests for v1.29.0 features:
- MCP-Native Tool Orchestration
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import (
    MCP_TRANSPORTS,
    MCPServerSpec,
    PlanSpec,
    TaskSpec,
)
from maestro_cli.runners import _build_mcp_config


def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return p


def _minimal_plan(*tasks: TaskSpec, **kwargs: object) -> PlanSpec:
    return PlanSpec(name="test", tasks=list(tasks), **kwargs)  # type: ignore[arg-type]


# ===========================================================================
# MCPServerSpec
# ===========================================================================


class TestMCPServerSpec:
    def test_defaults(self) -> None:
        spec = MCPServerSpec(name="test")
        assert spec.transport == "stdio"
        assert spec.timeout_sec == 30
        assert spec.command == []
        assert spec.description == ""
        assert spec.env == {}
        assert spec.allowed_task_roles == []
        assert spec.is_concurrency_safe is None

    def test_to_dict(self) -> None:
        spec = MCPServerSpec(
            name="github",
            command=["npx", "@modelcontextprotocol/server-github"],
            description="GitHub issues and pull requests",
            env={"GITHUB_TOKEN": "abc123"},
            allowed_task_roles=["qa-engineer"],
            is_concurrency_safe=False,
        )
        d = spec.to_dict()
        assert d["name"] == "github"
        assert d["command"] == ["npx", "@modelcontextprotocol/server-github"]
        assert d["description"] == "GitHub issues and pull requests"
        assert d["env"]["GITHUB_TOKEN"] == "abc123"
        assert d["allowed_task_roles"] == ["qa-engineer"]
        assert d["is_concurrency_safe"] is False

    def test_transports_constant(self) -> None:
        assert MCP_TRANSPORTS == {"stdio", "http", "sse"}


# ===========================================================================
# Loader — plan-level mcp_servers
# ===========================================================================


class TestMCPServerLoader:
    def test_parse_stdio_server(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "@modelcontextprotocol/server-github"]
                env:
                  GITHUB_TOKEN: "token123"
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [github]
        """)
        plan = load_plan(p)
        assert len(plan.mcp_servers) == 1
        assert plan.mcp_servers[0].name == "github"
        assert plan.mcp_servers[0].transport == "stdio"
        assert plan.mcp_servers[0].command == ["npx", "@modelcontextprotocol/server-github"]

    def test_parse_http_server(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: remote-tools
                transport: http
                url: "http://localhost:8080/mcp"
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [remote-tools]
        """)
        plan = load_plan(p)
        assert plan.mcp_servers[0].transport == "http"
        assert plan.mcp_servers[0].url == "http://localhost:8080/mcp"

    def test_parse_multiple_servers(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-github"]
              - name: jira
                command: ["npx", "server-jira"]
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [github, jira]
        """)
        plan = load_plan(p)
        assert len(plan.mcp_servers) == 2

    def test_parse_description_and_allowed_roles(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-github"]
                description: "GitHub repository and issue operations"
                allowed_task_roles: [qa-engineer, code-reviewer]
            tasks:
              - id: t1
                engine: claude
                agent: qa-engineer
                prompt: "analyze"
                mcp_tools: [github]
        """)
        plan = load_plan(p)
        server = plan.mcp_servers[0]
        assert server.description == "GitHub repository and issue operations"
        assert server.allowed_task_roles == ["qa-engineer", "code-reviewer"]

    def test_parse_is_concurrency_safe(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-github"]
                is_concurrency_safe: true
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [github]
        """)
        plan = load_plan(p)
        assert plan.mcp_servers[0].is_concurrency_safe is True

    def test_parse_is_concurrency_safe_camel_case_alias(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-github"]
                isConcurrencySafe: false
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [github]
        """)
        plan = load_plan(p)
        assert plan.mcp_servers[0].is_concurrency_safe is False

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - command: ["npx", "server-github"]
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        with pytest.raises(Exception, match="E069"):
            load_plan(p)

    def test_stdio_without_command_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: broken
                transport: stdio
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        with pytest.raises(Exception, match="E069"):
            load_plan(p)

    def test_http_without_url_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: broken
                transport: http
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        with pytest.raises(Exception, match="E069"):
            load_plan(p)

    def test_invalid_transport_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: broken
                transport: websocket
                command: ["npx", "server"]
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        with pytest.raises(Exception, match="E069"):
            load_plan(p)

    def test_duplicate_names_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-1"]
              - name: github
                command: ["npx", "server-2"]
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        with pytest.raises(Exception, match="E069"):
            load_plan(p)

    def test_no_mcp_servers(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        plan = load_plan(p)
        assert plan.mcp_servers == []


# ===========================================================================
# Loader — task-level mcp_tools
# ===========================================================================


class TestMCPToolsLoader:
    def test_parse_tools(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-github"]
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [github]
        """)
        plan = load_plan(p)
        assert plan.tasks[0].mcp_tools == ["github"]

    def test_unknown_tool_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
                mcp_tools: [nonexistent]
        """)
        with pytest.raises(Exception, match="E070"):
            load_plan(p)

    def test_role_filtered_tool_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            mcp_servers:
              - name: github
                command: ["npx", "server-github"]
                allowed_task_roles: [qa-engineer]
            tasks:
              - id: t1
                engine: claude
                agent: python-developer
                prompt: "analyze"
                mcp_tools: [github]
        """)
        with pytest.raises(Exception, match="allowed_task_roles"):
            load_plan(p)

    def test_no_tools(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "analyze"
        """)
        plan = load_plan(p)
        assert plan.tasks[0].mcp_tools == []


# ===========================================================================
# MCP Config Generation
# ===========================================================================


class TestBuildMCPConfig:
    def test_generates_config_file(self, tmp_path: Path) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", engine="claude", mcp_tools=["github"]),
            mcp_servers=[
                MCPServerSpec(
                    name="github",
                    command=["npx", "@mcp/server-github"],
                    env={"GITHUB_TOKEN": "tok"},
                ),
            ],
        )
        config_path = _build_mcp_config(plan, plan.tasks[0], tmp_path)
        assert config_path is not None
        assert config_path.exists()

        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert "mcpServers" in config
        assert "github" in config["mcpServers"]
        gh = config["mcpServers"]["github"]
        assert gh["command"] == "npx"
        assert gh["args"] == ["@mcp/server-github"]
        assert gh["env"]["GITHUB_TOKEN"] == "tok"

    def test_no_tools_returns_none(self, tmp_path: Path) -> None:
        plan = _minimal_plan(TaskSpec(id="t1", engine="claude"))
        result = _build_mcp_config(plan, plan.tasks[0], tmp_path)
        assert result is None

    def test_no_servers_returns_none(self, tmp_path: Path) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", engine="claude", mcp_tools=["foo"]),
        )
        result = _build_mcp_config(plan, plan.tasks[0], tmp_path)
        assert result is None

    def test_multiple_servers(self, tmp_path: Path) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", engine="claude", mcp_tools=["github", "jira"]),
            mcp_servers=[
                MCPServerSpec(name="github", command=["npx", "gh-server"]),
                MCPServerSpec(name="jira", command=["npx", "jira-server"]),
            ],
        )
        config_path = _build_mcp_config(plan, plan.tasks[0], tmp_path)
        assert config_path is not None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert len(config["mcpServers"]) == 2
        assert "github" in config["mcpServers"]
        assert "jira" in config["mcpServers"]

    def test_http_server_includes_url(self, tmp_path: Path) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", engine="claude", mcp_tools=["remote"]),
            mcp_servers=[
                MCPServerSpec(
                    name="remote",
                    url="http://localhost:8080",
                    transport="http",
                ),
            ],
        )
        config_path = _build_mcp_config(plan, plan.tasks[0], tmp_path)
        assert config_path is not None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["mcpServers"]["remote"]["url"] == "http://localhost:8080"

    def test_role_filtered_server_allowed_for_matching_agent(self, tmp_path: Path) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", engine="claude", agent="qa-engineer", mcp_tools=["github"]),
            mcp_servers=[
                MCPServerSpec(
                    name="github",
                    command=["npx", "gh-server"],
                    allowed_task_roles=["qa-engineer"],
                ),
            ],
        )
        config_path = _build_mcp_config(plan, plan.tasks[0], tmp_path)
        assert config_path is not None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert "github" in config["mcpServers"]


# ===========================================================================
# Serialization
# ===========================================================================


class TestMCPSerialization:
    def test_plan_to_dict_with_servers(self) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1"),
            mcp_servers=[MCPServerSpec(name="gh", command=["npx", "gh"])],
        )
        d = plan.to_dict()
        assert "mcp_servers" in d
        assert d["mcp_servers"][0]["name"] == "gh"

    def test_plan_to_dict_without_servers(self) -> None:
        plan = _minimal_plan(TaskSpec(id="t1"))
        d = plan.to_dict()
        assert "mcp_servers" not in d

    def test_task_to_dict_with_tools(self) -> None:
        t = TaskSpec(id="t1", mcp_tools=["github"])
        d = t.to_dict()
        assert d["mcp_tools"] == ["github"]

    def test_task_to_dict_without_tools(self) -> None:
        t = TaskSpec(id="t1")
        d = t.to_dict()
        assert d["mcp_tools"] == []
