# LangGraph Multi-Agent Team

Supervisor-pattern workflow with **Researcher**, **Coder**, and **Editor** specialists, a human-review breakpoint, NATS JetStream on both the outbound and inbound paths, Slack workers, Langfuse tracing (MLflow or LangSmith as alternatives), and LangSmith evaluations.

```
Shared State
     │
     ▼
Supervisor ──► Researcher / Coder / Editor ──► Supervisor
     │
     ▼
Review publisher ──► NATS (agent.review.required) ──► slack-publisher ──► Slack
     │
     ▼
[interrupt_before] Human review
     ▲
     │
workflow-resumer ◄── NATS (agent.review.feedback) ◄── slack-gateway ◄── Slack click
```

The graph never talks to Slack. The FastAPI webhook never talks to LangGraph. Each side publishes a JetStream event; a queue-group worker performs the I/O.

## Requirements

- Python 3.11+
- An OpenAI API key
- Optional: NATS with JetStream, Slack app, Langfuse (or MLflow, or LangSmith), Docker

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

If `from langchain_openai import ChatOpenAI` fails with `aiohttp.SocketTimeoutError` (common on mixed Anaconda environments), upgrade aiohttp in that environment: `pip install -U "aiohttp>=3.11"`.

## Run locally (terminal human review)

NATS is off by default (`NATS_ENABLED=false`), so the CLI can pause for review without a broker:

```bash
python -m agent_team.cli "Write a Python function that computes Fibonacci numbers."
```

When the graph pauses, type `APPROVE` or revision notes.

## LangGraph Server (Studio / API)

```bash
pip install -e ".[dev]" langgraph-cli
langgraph dev --host 127.0.0.1 --port 8123
```

The compiled graph is exported as `agent_team` from `langgraph.json`. Persistence is provided by the server (Postgres in Docker, in-memory for `langgraph dev` unless configured otherwise).

## Slack human review (via NATS)

1. Create a Slack app from `slack-app-manifest.yaml` and install it to your workspace.
2. Run NATS with JetStream, for example: `docker run --rm -p 4222:4222 -p 8222:8222 nats:2.11-alpine -js -m 8222`
3. Set `NATS_ENABLED=true`, `NATS_URL=nats://localhost:4222`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `SLACK_CHANNEL`, and `SLACK_ENABLED=true`.
4. Start the workers and the interactive gateway. NATS push-distributes each event to one member of each queue group:

```bash
python -m slack_publisher
python -m workflow_resumer
uvicorn slack_gateway.app:app --host 0.0.0.0 --port 8080
```

Scale either worker by running extra processes. If you previously ran the pull-consumer version, delete that durable once: `nats consumer rm AGENT_REVIEW slack-publisher`.

5. Expose `https://your-host/slack/interactive` as the Slack interactivity request URL.

Outbound: `review_publisher` writes `HumanReviewRequested` to `agent.review.required`; `slack_publisher` (queue `NATS_QUEUE`) posts the Block Kit card.

Inbound: the gateway writes `HumanReviewSubmitted` to `agent.review.feedback`; `workflow_resumer` (queue `NATS_FEEDBACK_QUEUE`) injects `feedback` into the paused thread and creates a new LangGraph run.

## Slack emulator (local HITL without Slack)

The emulator speaks Slack’s `chat.postMessage` API, then POSTs a signed Approve click to the gateway webhook so the graph can resume.

```bash
export SLACK_ENABLED=true
export SLACK_BOT_TOKEN=xoxb-emulator
export SLACK_SIGNING_SECRET=emulator-secret
export SLACK_API_BASE_URL=http://localhost:8081/api/
export SLACK_INTERACTIVITY_URL=http://localhost:8080/slack/interactive
python -m slack_emulator
```

Compose: `docker compose --profile slack-emulator up` with `SLACK_COMPOSE_API_BASE_URL=http://slack-emulator:8081/api/`. Cards accepted by the emulator are listed at `GET http://localhost:8081/reviews`.

## Docker

```bash
langgraph build -t agent-team:latest
IMAGE_NAME=agent-team:latest docker compose up --scale slack-publisher=3 --scale workflow-resumer=3
```

## Infrastructure telemetry (OpenTelemetry)

The Slack gateway and the NATS workers (`slack_publisher`, `workflow_resumer`) export OTLP spans. Collector URL, protocol, and auth are env vars so the same process can target Compose, a host collector, or a vendor intake.

```bash
# Compose default if unset: http://otel-collector:4318
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=slack-gateway
# OTEL_EXPORTER_OTLP_HEADERS=authorization=Bearer ...
```

Compose runs `otel-collector` (OTLP 4317/4318) and Jaeger (`http://localhost:16686`). Point `OTEL_COLLECTOR_FORWARD_ENDPOINT` at another OTLP backend when Jaeger is not the destination.

## Observability (Langfuse default)

Langfuse is the production tracer because it is MIT-licensed and self-hostable. MLflow is the experiment-platform alternative when the team already runs a tracking server. LangSmith is the LangChain-native fallback. See `src/agent_team/tracing.py` and `POST.md` §12.

```bash
# .env — Langfuse wins when several stacks are configured
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

```bash
# Alternative — MLflow, only when Langfuse keys are unset
# docker compose --profile mlflow up -d mlflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=agent-team
```

```bash
# Alternative — LangSmith, only when Langfuse keys and MLFLOW_TRACKING_URI are unset
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=multi-agent-slack-pipeline
```

Self-host Langfuse with the official compose (`docker compose -f docker-compose.prod.yml` in the [langfuse](https://github.com/langfuse/langfuse) repo). MLflow is the `mlflow` Compose profile in this project (`ghcr.io/mlflow/mlflow`, UI on port 5000). From LangGraph Server on the Compose network use `MLFLOW_TRACKING_URI=http://mlflow:5000`.

## Evaluations (LangSmith)

```bash
export USE_LOCAL_GRAPH=1
python evals/evaluate_agents.py
```

Evals set `auto_approve` so the human-review breakpoint does not stall CI. Keep `NATS_ENABLED=false` unless a broker is running. Metrics: code quality (LLM judge), routing efficiency, and cost/latency.

## Project layout

| Path | Role |
| --- | --- |
| `src/agent_team/` | Graph, state, specialist nodes, NATS publish, tracing |
| `src/review_events/` | Shared JetStream event contract |
| `src/slack_publisher/` | Queue worker: NATS → Slack |
| `src/slack_gateway/` | FastAPI webhook: Slack → NATS |
| `src/slack_emulator/` | Local Slack API stand-in that auto-approves via the gateway |
| `src/workflow_resumer/` | Queue worker: NATS → LangGraph resume |
| `src/telemetry/` | OpenTelemetry bootstrap for gateway and NATS workers |
| `otel-collector.yaml` | Collector receivers / exporters (env-substituted) |
| `evals/evaluate_agents.py` | LangSmith 3-metric harness |
| `langgraph.json` | LangGraph Server graph registration |
