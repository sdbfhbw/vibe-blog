# Security Policy

## Supported Versions

This project is currently in early development. Security fixes are applied to
the latest version on the default branch.

## Reporting a Vulnerability

Please do not open a public issue for suspected security vulnerabilities.

Report them privately to the project maintainers with:

- A clear description of the issue
- Steps to reproduce
- Impact assessment
- Affected files or endpoints, if known
- Any suggested mitigation

Until a dedicated security contact is published, use a private contact channel
for the repository owner or maintainer rather than public discussion.

## What to Report

Examples of relevant issues include:

- Leaked credentials or tokens
- Authentication or authorization bypasses
- Unsafe file upload or path traversal behavior
- Remote code execution or command injection
- Server-side request forgery
- Sensitive data exposure through logs, responses, or generated files

## Repository Hygiene

The repository intentionally excludes:

- Real `.env` files
- API keys and secrets
- Local databases
- Logs
- Generated outputs
- Dependency folders and virtual environments

Use `backend/.env.example` for configuration examples and keep all real secrets
outside version control.

## Response Expectations

Maintainers should aim to:

1. Acknowledge reports promptly.
2. Confirm impact and affected versions.
3. Prepare a fix or mitigation.
4. Coordinate disclosure after a safe update path exists.
