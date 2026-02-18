# Security Policy

AgentField takes security seriously. We appreciate responsible disclosure by the security
research community and commit to working with you to understand, remediate, and communicate
security vulnerabilities promptly.

## Supported Versions

We release security patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

Older versions receive no security backports. We strongly encourage all users to run the
latest 0.1.x release.

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.** Public disclosure before a
fix is available puts all AgentField users at risk.

Please report vulnerabilities by emailing **contact@agentfield.ai** with the subject line
`[SECURITY] <short description>`. You may use the report template in the
[What to Include in Your Report](#what-to-include-in-your-report) section below.

PGP-encrypted email is welcome; contact us at the address above for our public key.

## Responsible Disclosure Process

We follow a coordinated disclosure model:

| Step | Timeline | Description |
|------|----------|-------------|
| **Acknowledgement** | Within 48 hours of receipt | We confirm that your report has been received and assign an internal tracking ID. |
| **Triage** | Within 5 business days of acknowledgement | We assess severity (using CVSS v3.1), confirm reproducibility, and communicate our initial findings. |
| **Fix & Release** | Within 90 days of acknowledgement | We develop, test, and release a patch. Complex or systemic issues may require more time; we will communicate any extension. |
| **Public Disclosure** | At fix release, or at 90-day window close | We publish an advisory (GitHub Security Advisory + CHANGELOG entry). Earlier disclosure by mutual agreement is welcome. |

**Embargo request:** We ask that you do not publicly disclose details of the vulnerability —
including proof-of-concept code, exploit techniques, or affected version ranges — until the fix
ships or the 90-day window closes, whichever comes first. We will notify you before any public
statement so you can coordinate timing.

## Scope

### In-Scope

The following assets are in scope for responsible disclosure:

- **Control plane** (`control-plane/`) — the AgentField server, API, and orchestration logic
- **Go SDK** (`sdk/go/`)
- **Python SDK** (`sdk/python/`)
- **TypeScript SDK** (`sdk/typescript/`)
- **Docker and Kubernetes deployment configurations** (`deployments/`)

### Out-of-Scope

The following are **not** in scope:

- **Third-party dependencies** — vulnerabilities in upstream libraries we consume (report those
  to the upstream maintainer; we will track and patch our own upgrade)
- **Denial-of-service attacks** without demonstrated security impact beyond availability
- **Social engineering** of project maintainers or contributors
- **Physical attacks** against infrastructure

If you are unsure whether a finding is in scope, please report it — we would rather evaluate
an out-of-scope report than miss a real vulnerability.

## What to Include in Your Report

A high-quality report helps us triage faster. Please include:

1. **Affected component and version** — e.g., `control-plane v0.1.40`, `sdk/python v0.1.38`
2. **Vulnerability description** — what the flaw is and what security property it violates
   (confidentiality, integrity, availability)
3. **Steps to reproduce** — a minimal, reliable sequence of steps or a proof-of-concept script
4. **Potential impact** — who can exploit this, under what conditions, and what they can achieve
5. **Suggested fix** *(optional)* — if you have a recommended remediation, we welcome it

Incomplete reports are still welcome — share what you have and we will follow up.

## Researcher Recognition

We believe in recognising the work of security researchers who help make AgentField safer.

Reporters of **valid, in-scope vulnerabilities** will be credited by name (or handle) in the
release changelog and the associated GitHub Security Advisory, unless you request anonymity.
Please indicate your preferred credit name in your initial report.

We do not currently offer a monetary bug-bounty programme, but we acknowledge your contribution
publicly and, where appropriate, in project release notes.

## Safe Harbour

AgentField project maintainers will not pursue legal action against security researchers who:

- Discover and report vulnerabilities in good faith under this policy
- Avoid accessing, modifying, or deleting data beyond what is necessary to demonstrate the
  vulnerability
- Do not exploit a vulnerability beyond the minimum required to confirm its existence
- Do not conduct research that causes degraded service for other users
- Comply with all applicable laws throughout their research

We consider good-faith security research conducted under this policy to be authorised, and we
will not refer such research to law enforcement. If a third party initiates legal action against
a researcher who has followed this policy, we will make clear that the research was conducted
with our knowledge and in accordance with this policy.

---

*This policy is effective 2026-02-18 and supersedes all prior security disclosure guidance.*
