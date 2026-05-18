<div align="center">

# vibe-blog

_Turn complex tech into stories everyone can understand._

**[中文](README.md) | English**

[![Version](https://img.shields.io/badge/version-v0.1.0-4CAF50.svg)](https://github.com/sdbfhbw/vibe-blog)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)

</div>

vibe-blog is a multi-agent long-form content generation project that connects retrieval, planning, drafting, code generation, illustration generation, review, and export into one runnable workflow.

## Core Capabilities

- Multi-agent workflow for research, planning, drafting, review, and assembly
- Retrieval augmentation with source filtering and citation preparation
- Long-form article generation with code blocks, Mermaid diagrams, and AI-generated illustrations
- Frontend and backend integration with live progress, Markdown preview, and export support
- Additional modules for retrieval, tutorial review, visual content, and publishing workflows

## Tech Stack

### Backend

- Python 3.10+
- Flask
- LangGraph
- Jinja2
- Server-Sent Events

### Frontend

- Vue 3
- Vite
- TypeScript
- Mermaid
- Vitest

### Related Services

- OpenAI-compatible model APIs
- Zhipu Search
- Image generation service
- Langfuse tracing

## Quick Start

### Option 1: Docker

```bash
cp backend/.env.example backend/.env
docker compose -f docker/docker-compose.yml up -d
```

After startup:

- Frontend: `http://localhost:3000`
- API: `http://localhost:5000`

### Option 2: Local Development

1. Clone the repository

```bash
git clone https://github.com/sdbfhbw/vibe-blog
cd vibe-blog
```

2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
pip install -r requirements-dev.txt
npm run install:frontend
```

3. Configure environment variables

```bash
cp backend/.env.example backend/.env
```

4. Start the backend

```bash
cd backend
python app.py
```

5. Start the frontend

```bash
cd frontend
npm run dev
```

Default local endpoints:

- Frontend: `http://localhost:5173`
- API: `http://localhost:5001/api`

## Environment Variables

See [backend/.env.example](./backend/.env.example) for the full configuration. Common variables:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI-compatible API key |
| `OPENAI_API_BASE` | OpenAI-compatible API base URL |
| `TEXT_MODEL` | Text generation model |
| `ZAI_SEARCH_API_KEY` | Zhipu Search API key |
| `NANO_BANANA_API_KEY` | Image generation service key |
| `TRACE_ENABLED` | Enable Langfuse tracing |

## Project Structure

```text
vibe-blog/
├── backend/
│   ├── app.py                         # Flask application entrypoint
│   ├── routes/                        # API routes
│   ├── services/
│   │   ├── blog_generator/            # Core multi-agent blog workflow
│   │   │   ├── agents/                # Writer, Planner, Reviewer, and other agents
│   │   │   ├── orchestrator/          # Workflow orchestration
│   │   │   ├── services/              # Search, crawling, and source filtering
│   │   │   ├── tools/                 # Agent tool wrappers
│   │   │   └── workflow_configs/      # Workflow configuration
│   │   ├── chat/                      # Chat and writing sessions
│   │   ├── publishers/                # Publishing adapters
│   │   └── task_queue/                # Scheduled jobs and queues
│   ├── eval/                          # Retrieval evaluation scripts and datasets
│   ├── tests/                         # Backend tests
│   └── vibe_reviewer/                 # Tutorial quality review module
├── frontend/
│   ├── src/
│   │   ├── components/                # UI components
│   │   ├── composables/               # Reusable composition logic
│   │   ├── services/                  # Frontend API wrappers
│   │   ├── stores/                    # State management
│   │   └── views/                     # Route-level views
│   └── __tests__/                     # Frontend tests
├── docker/                            # Docker deployment files
├── docs/                              # Design and deployment notes
├── tests/                             # End-to-end and cross-module tests
├── requirements.txt                   # Backend runtime dependency entrypoint
├── requirements-dev.txt               # Backend dev/test dependency entrypoint
└── package.json                       # Frontend command entrypoint
```

## Tests

```bash
# backend
pytest

# frontend
npm run test:frontend
```

## Related Docs

- [CHANGELOG.md](./CHANGELOG.md)
- [TESTING.md](./TESTING.md)
- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [SECURITY.md](./SECURITY.md)

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
