# Contributing to vibe-blog

Thanks for contributing to vibe-blog. This project combines a Flask backend,
a Vue frontend, and optional third-party AI services, so useful contributions
usually need both code discipline and clear verification notes.

## Before You Start

1. Search existing issues and pull requests before opening a new one.
2. Keep changes focused. Prefer one concern per pull request.
3. Do not commit local runtime files, generated outputs, credentials, caches,
   virtual environments, or dependency folders.
4. Use `backend/.env.example` as the public configuration reference. Never
   commit a real `.env` file or API key.

## Local Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
npm run install:frontend
cp backend/.env.example backend/.env
```

Start the applications in separate terminals:

```bash
cd backend
python app.py
```

```bash
cd frontend
npm run dev
```

## Development Guidelines

- Follow the existing module boundaries and naming conventions.
- Prefer small, reviewable refactors over broad rewrites.
- Keep user-facing behavior stable unless the pull request is explicitly about
  changing it.
- Add or update tests for new behavior and bug fixes.
- Keep docs aligned with the actual commands, ports, and configuration keys.
- For generated content or screenshots, include only assets that are necessary
  for documentation or tests.

## Verification

Run the checks relevant to your change before opening a pull request.

```bash
# Backend
pytest

# Frontend
npm run test:frontend
npm run build:frontend
```

If you cannot run a check, state exactly why in the pull request description.

## Pull Request Checklist

- [ ] The change is focused and explained clearly.
- [ ] Tests were added or updated where behavior changed.
- [ ] Relevant documentation was updated.
- [ ] No secrets, local databases, logs, generated outputs, or build artifacts
      were committed.
- [ ] The pull request description lists verification performed and any known
      gaps.

## Reporting Bugs

Useful bug reports include:

- Expected behavior
- Actual behavior
- Minimal reproduction steps
- Relevant logs or screenshots with secrets removed
- OS, Python version, Node version, and whether Docker or local development was
  used

## License

By contributing, you agree that your contributions are provided under the
repository license.
