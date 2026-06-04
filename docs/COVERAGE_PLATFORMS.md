# Coverage & Quality Platforms

Maestro uploads the same CI-generated coverage to several services for
redundant validation and badges. All wiring is already in the workflows; each
service is **gated so CI stays green until you enable it**. Coverage artifacts
(`coverage.xml`, `coverage.lcov`) are produced once on the `ubuntu-latest / 3.12`
test lane and reused by every uploader.

| Service | Token needed | Where it runs | Status |
|---------|--------------|---------------|--------|
| **Codecov** | none (public) | `ci.yml` test job | live (badge in README) |
| **Coveralls** | none (public) | `ci.yml` test job | live (badge in README) |
| **SonarCloud** | `SONAR_TOKEN` | `sonarcloud.yml` | enable below |
| **Codacy** | `CODACY_PROJECT_TOKEN` | `ci.yml` test job | enable below |

Codecov and Coveralls need no token on public repos — for Coveralls, just
sign in at https://coveralls.io with GitHub and toggle this repo on; the badge
then turns from "unknown" to the real number on the next push to `main`.

## SonarCloud

1. Sign in at https://sonarcloud.io with GitHub and import `maestro-cli`.
2. Project **Administration → Analysis Method → disable "Automatic Analysis"**
   (it conflicts with the CI scan).
3. Generate a token (Account → Security) and add it as repo secret `SONAR_TOKEN`
   (Settings → Secrets and variables → Actions → New repository secret).
4. If SonarCloud picks a different organization/project key, update
   `sonar-project.properties` and the badge below to match.

Badge (paste into the README badge block once live):

```markdown
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=tiagojcperez_maestro-cli&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=tiagojcperez_maestro-cli)
[![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=tiagojcperez_maestro-cli&metric=coverage)](https://sonarcloud.io/summary/new_code?id=tiagojcperez_maestro-cli)
```

## Codacy

1. Sign in at https://app.codacy.com with GitHub and add `maestro-cli`.
2. Project Settings → Coverage → copy the **Project API token**; add it as repo
   secret `CODACY_PROJECT_TOKEN`.
3. The `ci.yml` "Upload coverage to Codacy" step activates automatically on the
   next push.

Badge: copy the ready-made markdown from Codacy (Settings → Badges) — it embeds
your project UUID — and paste it into the README badge block. It looks like:

```markdown
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/REPLACE_WITH_YOUR_UUID)](https://app.codacy.com/gh/tiagojcperez/maestro-cli/dashboard)
```
