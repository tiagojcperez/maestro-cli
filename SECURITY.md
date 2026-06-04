# Security Policy

This document is the coordinated vulnerability disclosure policy for Maestro
CLI (`maestro-ai-cli`). For the architectural threat model and the mitigations
Maestro ships, see [docs/SECURITY.md](docs/SECURITY.md). For the operational
hardening checklist, see [docs/SECURITY_BASELINE.md](docs/SECURITY_BASELINE.md).

## Supported Versions

Maestro CLI is on the `2.x` release line. Security fixes are applied to the
latest released `2.x` version.

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | Yes (latest patch) |
| 1.x     | No                 |
| < 1.0   | No                 |

If you are running an older release, upgrade to the latest `2.x` before
reporting an issue, in case the problem is already fixed.

## How to Report a Vulnerability

Please report security vulnerabilities **privately** through GitHub's private
vulnerability reporting (Security Advisories):

- https://github.com/tiagojcperez/maestro-cli/security/advisories

Click **"Report a vulnerability"** to open a private advisory. This keeps the
report confidential while it is being assessed and fixed.

**Do NOT open a public GitHub issue, pull request, or discussion for a security
vulnerability.** Public disclosure before a fix is available puts users at
risk. Use the private advisory channel above instead.

When reporting, please include as much of the following as you can:

- A clear description of the issue and its impact.
- The affected component (for example: plan loader, engine subprocess
  handling, context pipeline, secret masking, plugin loading).
- Steps to reproduce, ideally a minimal plan YAML or command sequence.
- The Maestro version (`maestro --version`), Python version, and OS.
- Any relevant logs or run artifacts, with secrets removed.

## Response Expectations

Maestro CLI is a solo-maintained, local-first project. There is no dedicated
security team and no paid bug-bounty program. Response is best-effort:

- **Acknowledgement**: typically within a few days of a valid report.
- **Assessment and fix**: prioritized by severity and exploitability; timelines
  vary with maintainer availability.
- **Disclosure**: coordinated with the reporter. A fix and advisory are
  published once a patch is available. Credit is given to reporters who want it.

Please allow reasonable time for a fix before any public disclosure.

## Scope

In scope:

- The `maestro` CLI and the `maestro_cli` Python package in this repository.
- The documented plan schema, run artifacts, and CLI behavior.
- Issues that let untrusted input (upstream task output, machine-generated
  plans, MCP tool documentation, external data) escalate beyond its intended
  trust boundary, leak declared secrets, or bypass documented safety controls.

Out of scope:

- Vulnerabilities in the underlying engine CLIs (`codex`, `claude`, `gemini`,
  `copilot`, `qwen`, `ollama`, `llama-cli`) or their model providers. Report
  those to the respective vendors.
- Vulnerabilities in third-party dependencies. Report those upstream; we will
  update the pin once a fixed version is available.
- Arbitrary code execution that requires the operator to deliberately author a
  dangerous plan, install an untrusted engine plugin, or pass yolo/bypass flags.
  These are documented trust decisions, not vulnerabilities. See
  [docs/SECURITY.md](docs/SECURITY.md).
- Cost, rate-limit, or quota exhaustion caused by an operator's own plan running
  against their own provider account.

## A Note on Maestro's Trust Model

Maestro orchestrates AI engine CLIs that you trust and run locally with your
own credentials. It ships defense-in-depth controls for *untrusted data*
(context taint tracking, observation sandboxing, semantic firewall, output
redaction, the `SEC001`-`SEC023` audit rules, and a runtime policy engine), but
it does not sandbox the *engines themselves* or the *plans you author*. Treat
authored plans and installed plugins as trusted code. See
[docs/SECURITY.md](docs/SECURITY.md) for the full boundary analysis.
