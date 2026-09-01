# Building a Production Multi-Agent Workflow with LangGraph, Human Review, and Slack

Linear LLM chains are enough for a one-shot Q&A bot. They fall apart as soon as the work is cyclic: research, then code, then review, then a human says “not quite”, then the team loops again. That is the gap LangGraph is designed to fill.

This article walks through a complete implementation of a **supervisor-pattern multi-agent system** with three specialists (Researcher, Coder, Editor), a **durable human-in-the-loop breakpoint**, **NATS JetStream** as the notification bus, a **Slack delivery microservice**, a **FastAPI webhook** that resumes the graph, **OpenTelemetry** on the webhook and NATS workers, **Langfuse tracing** (with MLflow or LangSmith as opt-in alternatives), and a LangSmith evaluation harness. The stack uses current library versions: LangGraph 1.2, langchain-openai 1.6, nats-py, Pydantic v2, FastAPI, OpenTelemetry, Langfuse, MLflow, and LangGraph Server with Postgres and Redis.

The sections below explain *why* each piece exists and *how* to assemble it.

---

## 1. Why LangGraph instead of a LangChain chain

LangChain is excellent at directed acyclic pipelines: retrieve, stuff context, generate, return. State is implicit, and control flow moves forward.

LangGraph treats the application as a **state machine**. Every node reads and writes a shared state object. Edges can loop. Execution can pause, persist a checkpoint, and resume days later on a different process. That combination is what makes multi-agent coordination and human review practical rather than bolted on.

| Concern | LangChain chain | LangGraph |
| --- | --- | --- |
| Structure | DAG / sequence | Cyclic graph |
| State | Passed along the chain | Central, typed, persistent |
| Human-in-the-loop | Custom wrappers | Native `interrupt_before` / `interrupt()` |
| Debugging | Stack traces | Checkpoints and time travel |
| Fit | RAG, simple Q&A | Agent teams, long-running jobs |

If the product needs a researcher, a coder, an editor, and a human sign-off that can reject the artifact, LangGraph is the right layer.

---

## 2. Architecture: supervisor, specialists, breakpoint

The design is a **supervisor pattern**. A router LLM inspects the shared state and chooses the next specialist. Specialists write their artifacts back into state and return to the supervisor. When the team is ready, the graph **publishes a review event to NATS JetStream** and **pauses before** the human-review node. A separate worker consumes that stream and is the only process that talks to Slack. NATS is the messaging layer that keeps those processes decoupled; the review-publishing section below defines streams and workers before walking through the code.

```
                  +--------------------------------+
                  |         Shared State           |
                  | messages, notes, code, status  |
                  +---------------+----------------+
                                  |
                                  v
                      +-----------+-----------+
                      |   Supervisor / Router |
                      +-----+-----+-----+-----+
                            |     |     |
         +------------------+     |     +------------------+
         |                        |                        |
         v                        v                        v
  +------+-------+        +-------+------+         +-------+------+
  |  Researcher  |        |    Coder     |         |    Editor    |
  +------+-------+        +-------+------+         +-------+------+
         |                        |                        |
         +------------------+     |     +------------------+
                            |     |     |
                            v     v     v
                      +-----------+-----------+
                      |  Review publisher     |
                      |  (NATS JetStream)     |
                      +-----------+-----------+
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
      +-----------+-----------+       +-----------+-----------+
      |  [interrupt_before]   |       | slack-publisher       |
      |   Human Review Node   |       | (Slack Block Kit)     |
      +-----------------+-----+       +-----------------------+
                        |
                   +----+------------------------+
                   |                             |
      (Approved)   v                             v (Feedback)
          +--------+--------+          +---------+---------+
          |  Finish         |          | Back to Supervisor|
          +-----------------+          +-------------------+
```

Two details matter in production:

1. **Publish, then pause.** The NATS event must be written *before* the interrupt so the Slack worker can notify reviewers while the graph is frozen.
2. **Keep Slack out of the graph.** The graph only publishes `HumanReviewRequested`. Delivery, retries, and Slack API errors belong in `slack_publisher`.
3. **Resume into the review node, do not skip it.** The review node is what maps “APPROVE” versus revision text onto `finish` or `supervisor`. Updating state *as if* that node already ran skips the decision and can trap the graph.

---

## 3. Project layout

Keep the graph, the notification bus, the Slack worker, the Slack gateway, and the evals as separate packages. LangGraph Server only needs to import a compiled graph; the NATS subscriber and webhook are different processes.

```
src/agent_team/          # graph, state, nodes, CLI, NATS publish node
src/review_events/       # shared JetStream event contract
src/slack_publisher/     # microservice: NATS → Slack
src/slack_gateway/       # FastAPI interactivity endpoint
src/workflow_resumer/    # microservice: NATS → LangGraph resume
src/telemetry/           # OpenTelemetry bootstrap for gateway and workers
evals/evaluate_agents.py
langgraph.json           # Server graph registration
docker-compose.yml
otel-collector.yaml
slack-app-manifest.yaml
```

Package the Python code with `pyproject.toml` and a `src/` layout so `pip install -e .` and `langgraph.json` `"dependencies": ["."]` both work. Pin current majors, for example `langgraph>=1.2,<2` and `langchain-openai>=0.3,<2`.

Configuration belongs in environment variables, loaded with `pydantic-settings`. At minimum you need `OPENAI_API_KEY` and `OPENAI_MODEL` (this implementation defaults to `gpt-4o`). Slack, OpenTelemetry, Langfuse (or MLflow, or LangSmith), and the LangGraph Server URL are optional until you turn those layers on. Hostnames for NATS, LangGraph, and the OTLP collector are env vars as well: the same process image must be able to run in Compose, on a laptop, or on a remote host without a code change.

---

## 4. Shared state

Every node returns a partial dictionary that LangGraph merges into `AgentState`. Messages use the `add_messages` reducer so history accumulates. Artifact fields overwrite: the latest research notes, code, and editor comments win.

```python
from typing import Annotated, Literal, NotRequired, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    research_notes: NotRequired[str]
    current_code: NotRequired[str]
    editor_notes: NotRequired[str]
    feedback: NotRequired[str]
    next_action: NotRequired[str]
    step_count: NotRequired[int]
```

`NotRequired` lets the first `invoke` send only `messages`. `step_count` is a safety valve: if the supervisor loops too many times, force human review instead of burning tokens forever.

A small helper should recover the **original user request** from the first human message, not the last item in `messages`. After a few specialist turns the last message is editor commentary, which is a poor research prompt.

---

## 5. Specialist nodes

Each specialist is a plain function: `(state) -> dict`. No tools, no inner agents. That keeps the outer graph easy to interrupt, test, and trace.

**Researcher** turns the user request plus any human feedback into concise technical notes.

**Coder** writes Python from those notes, the existing artifact, and feedback. Strip markdown fences so `current_code` is source, not a fenced blob.

**Editor** reviews the code and stores `editor_notes` as an explicit field. Do not rely on “the last message” for Slack; an explicit field survives field masks and later nodes.

```python
def coder_node(state: AgentState) -> dict:
    prompt = (
        f"User request:\n{original_request(state)}\n\n"
        f"Research notes:\n{state.get('research_notes')}\n\n"
        f"Existing code:\n{state.get('current_code') or ''}\n\n"
        f"Feedback to address:\n{state.get('feedback') or 'None'}"
    )
    response = get_llm().invoke(
        [SystemMessage(content=CODER_SYSTEM), HumanMessage(content=prompt)]
    )
    return {
        "current_code": strip_code_fences(str(response.content)),
        "messages": [response],
    }
```

Instantiate the LLM inside a factory (`ChatOpenAI(model=..., temperature=0)`), not at import time. That keeps unit tests and LangGraph Server startup from requiring a live OpenAI client.

---

## 6. Supervisor routing with structured output

The supervisor is the only node that *decides*. Give it a Pydantic schema and OpenAI’s JSON-schema structured output so `next_step` is a closed set of names, not free text you have to parse.

```python
class Router(BaseModel):
    next_step: Literal["researcher", "coder", "editor", "human_review"]
    reason: str

def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    step_count = int(state.get("step_count") or 0) + 1
    if step_count >= get_settings().max_supervisor_steps:
        return {"next_action": "human_review", "step_count": step_count}

    llm = get_llm().with_structured_output(Router, method="json_schema")
    decision = llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM), HumanMessage(content=brief), *messages]
    )
    return {"next_action": decision.next_step, "step_count": step_count}
```

The system prompt should encode a sensible policy: research first, then code, then edit, then human review; on rejection, send work back to the coder (or researcher if the feedback is about requirements). Pass a **state brief**—notes, code, feedback, step count—not only chat history. The router cannot choose well if it cannot see the artifacts.

Map `next_action` with `add_conditional_edges`. When the supervisor picks `human_review`, route to `review_publisher` first, then to `human_review`.

---

## 7. Compile the graph and pause before human review

```python
workflow = StateGraph(AgentState)
# add nodes and edges ...
workflow.add_edge("review_publisher", "human_review")
workflow.add_conditional_edges(
    "human_review",
    route_after_human,
    {"finish": END, "supervisor": "supervisor"},
)

graph = workflow.compile(interrupt_before=["human_review"])
```

`interrupt_before=["human_review"]` is a **static breakpoint**. Execution runs `review_publisher` (NATS publish), then writes a checkpoint and stops. The next node listed on the snapshot is `human_review`, but that function has not run yet.

Two compile modes:

- **LangGraph Server / Platform:** compile *without* a checkpointer. The server injects Postgres persistence.
- **Local CLI:** compile with `InMemorySaver()` so a `thread_id` can pause and resume in one process.

Export a module-level `graph` (an `app` alias is harmless if other tooling looks for that name) so `langgraph.json` can point at it:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "python_version": "3.11",
  "dependencies": ["."],
  "graphs": {
    "agent_team": {
      "path": "./src/agent_team/graph.py:graph",
      "description": "Supervisor team with human review."
    }
  },
  "env": ".env"
}
```

The human-review node itself is small: if feedback contains an approval token (`APPROVE`, `APPROVED`, `LGTM`), set `next_action` to `finish`; otherwise send the team back to the supervisor. Support `configurable.auto_approve` so evaluations and CI do not hang on a missing human.

```python
def human_review_node(state: AgentState, config: RunnableConfig) -> dict:
    auto_approve = bool((config.get("configurable") or {}).get("auto_approve"))
    feedback = "APPROVE" if auto_approve else (state.get("feedback") or "")
    if is_approved(feedback):
        return {"feedback": feedback, "next_action": "finish"}
    return {"feedback": feedback or "Changes requested.", "next_action": "supervisor"}
```

`route_after_human` must return the **string** `"finish"` (mapped to `END`) or `"supervisor"`. Returning the `END` constant from the router while mapping the key `"finish"` is a common mismatch.

---

## 8. Local human-in-the-loop (terminal)

Before Slack, prove the breakpoint with a CLI. Invoke once, then inspect `graph.get_state(config).next`.

If `next` includes `human_review`, print the artifact, read stdin, `update_state` with `{"feedback": ...}`, and `invoke(None, config)` to continue. `None` means “resume from the checkpoint,” not “start a new run.”

```python
config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}
graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)

while True:
    snapshot = graph.get_state(config)
    if not snapshot.next:
        break
    if "human_review" in snapshot.next:
        feedback = input("APPROVE or describe changes: ").strip()
        graph.update_state(config, {"feedback": feedback})
        graph.invoke(None, config)
        continue
    graph.invoke(None, config)
```

Every invoke that uses a checkpointer **must** pass the same `thread_id`. Without it, LangGraph cannot load the paused checkpoint.

---

## 9. Publish the review to NATS, deliver Slack from a worker

### What NATS is, and why streams and workers

[NATS](https://nats.io) is a lightweight publish/subscribe messaging system. Producers send messages to a **subject** (a named topic such as `agent.review.required`). Consumers subscribe to that subject. Neither side needs to know the other’s host, language, or uptime. That is the whole point here: the LangGraph process should not import Slack, and the Slack webhook should not import LangGraph.

Core NATS is fire-and-forget. If `slack_publisher` is restarting when the graph publishes, the notification is gone and the human never sees the card. **JetStream** is NATS’s persistence layer. A **stream** is an append-only log bound to one or more subjects. Messages stay in the stream until a consumer acknowledges them. The server can redeliver, deduplicate by `Nats-Msg-Id`, and replay after a crash. That matches a human-in-the-loop breakpoint: the graph has already paused; losing the outbound review or the inbound Slack click would leave the thread stuck forever.

A **worker** in this design is a JetStream **queue-group** consumer: several processes bind to the same durable group, and the server delivers each message to exactly one of them. That is competing consumers, not a fan-out. Scale `slack_publisher` to three replicas and you still post one Slack card. Scale `workflow_resumer` and you still resume each thread once. The worker `ack`s after a successful Slack post or LangGraph resume, and `nak`s on a retryable failure so the stream hands the message to any member of the group.

Those three ideas — subject, stream, queue-group worker — are why the code looks the way it does: `review_publisher` only writes to JetStream, `ensure_stream` creates the log if it is missing, and the side processes are workers rather than extra nodes inside the graph.

The graph must not call Slack. When the supervisor decides the artifact is ready, `review_publisher` serializes a `HumanReviewRequested` event and publishes it to JetStream subject `agent.review.required`. The payload includes everything the reviewer needs:

- `thread_id` (LangGraph thread, also used as Slack button `value`)
- original request, research notes, current code, editor notes
- prior human feedback, step count, recent messages
- `event_id` for JetStream deduplication (`Nats-Msg-Id`)

```python
async def review_publisher_node(state: AgentState, config: RunnableConfig) -> dict:
    event = build_review_event(state, config)
    if not settings.nats_enabled:
        return {}
    async with jetstream_connection(settings.nats_url) as (_nc, js):
        await ensure_stream(js, stream=settings.nats_stream, subject=settings.nats_subject)
        await publish_event(js, subject=settings.nats_subject, event=event)
    return {}
```

`slack_publisher` is a **push** consumer on JetStream queue group `slack-publisher` (env `NATS_QUEUE`). Every instance binds to the same durable deliver group; the server sends each message to one worker. nats-py requires the queue name and durable name to be identical. The callback validates JSON, posts Block Kit, then `ack`s. Slack API failures `nak` so JetStream redelivers to any member of the group. Invalid payloads are acked and logged so they do not block the queue.

If `NATS_ENABLED=false`, the node logs and returns `{}` so the CLI can still pause without a broker. Truncate only in the Slack card (~800 characters of code); the NATS event keeps the full artifact.

Create the Slack app from a manifest (`slack-app-manifest.yaml`): bot user, interactivity enabled, `chat:write` and `channels:join`. Point `request_url` at a **public HTTPS** URL that reaches `POST /slack/interactive`.

For local runs without a Slack workspace, a **Slack emulator** implements `chat.postMessage` the same way `slack_sdk.WebClient` calls it. `slack_publisher` posts the Block Kit card there (`SLACK_API_BASE_URL`). The emulator returns Slack’s `{ok: true, ts, channel}` shape, then after a short delay POSTs a signed `block_actions` body — Approve, `feedback=APPROVE` — to `SLACK_INTERACTIVITY_URL` (the gateway). The HMAC uses `SLACK_SIGNING_SECRET`, so the gateway cannot tell the click from a real one. The rest of the path (NATS feedback, `workflow_resumer`) is unchanged.

```bash
export SLACK_ENABLED=true
export SLACK_BOT_TOKEN=xoxb-emulator
export SLACK_SIGNING_SECRET=emulator-secret
export SLACK_API_BASE_URL=http://localhost:8081/api/
export SLACK_INTERACTIVITY_URL=http://localhost:8080/slack/interactive
python -m slack_emulator
```

On the Compose network use `SLACK_COMPOSE_API_BASE_URL=http://slack-emulator:8081/api/` and `SLACK_INTERACTIVITY_URL_COMPOSE=http://slack-gateway:8080/slack/interactive`, then `docker compose --profile slack-emulator up`. Inspect accepted cards at `GET /reviews`.

---

## 10. FastAPI gateway: queue Slack clicks onto NATS

Slack interactivity is a form POST whose `payload` field is JSON. The gateway must stay fast (Slack’s 3-second timeout) and must **not** call LangGraph. It only verifies the request, maps the click to a `HumanReviewSubmitted` event, and publishes it to `agent.review.feedback`.

1. Verify `X-Slack-Signature` / `X-Slack-Request-Timestamp` against the raw body (HMAC-SHA256, five-minute skew).
2. Parse `actions[0]` — `actions` is a **list**, not a dict.
3. Map `approve_btn` to `feedback="APPROVE"` and `reject_btn` to the plain-text input under `state.values`.
4. Publish the event (thread id, action, feedback, Slack user/channel context) to JetStream.
5. Return an ephemeral acknowledgement.

```python
@app.post("/slack/interactive")
async def slack_interactive(request: Request):
    raw_body = await request.body()
    if not verify_slack_signature(...):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")
    payload = json.loads(str((await request.form()).get("payload")))
    decision = parse_review_payload(payload)
    event = decision_from_slack(decision, payload, assistant_id="agent_team")
    await publish_feedback_event(event, settings)
    return {"response_type": "ephemeral", "text": "Choice submitted."}
```

`workflow_resumer` is a **push** consumer on queue group `workflow-resumer` (`NATS_FEEDBACK_QUEUE`). Scale it the same way as `slack_publisher`. Each instance injects feedback into the paused checkpoint and starts a new run:

```python
await client.threads.update_state(
    thread_id=event.thread_id,
    values={"feedback": event.feedback},
)
await client.runs.create(
    thread_id=event.thread_id,
    assistant_id=event.assistant_id,
)
```

Do **not** pass `as_node="human_review"` here. That tells LangGraph “pretend human_review already executed,” which skips the node that converts feedback into `finish` versus `supervisor`. The interrupt left `human_review` as the next step; injecting `feedback` and creating a new run is enough for that node to execute.

Use `langgraph_sdk.get_client(url=LANGGRAPH_SERVER_URL)` **only** inside `workflow_resumer`. The assistant id must match the key in `langgraph.json` (`agent_team`).

---

## 11. LangGraph Server and Docker

Local development:

```bash
pip install -e ".[dev]" langgraph-cli
cp .env.example .env   # set OPENAI_API_KEY
langgraph dev --host 127.0.0.1 --port 8123
```

Studio and the SDK then talk to `http://localhost:8123`. For Slack delivery, run JetStream and the side processes:

```bash
docker run --rm -p 4222:4222 -p 8222:8222 nats:2.11-alpine -js -m 8222
python -m slack_publisher
python -m workflow_resumer
uvicorn slack_gateway.app:app --host 0.0.0.0 --port 8080
```

Production-like hosting needs **Postgres** (checkpoints), **Redis** (the API job queue), and **NATS JetStream** (review events). The official LangGraph API image listens on container port `8000`; map it to `8123` on the host. Pass `DATABASE_URI`, `REDIS_URI`, and `NATS_URL`. Build the application image with `langgraph build`, then compose:

```bash
langgraph build -t agent-team:latest
IMAGE_NAME=agent-team:latest docker compose up --scale slack-publisher=3 --scale workflow-resumer=3
```

A typical compose file includes:

- `nats:2.11-alpine` with `-js -m 8222`
- `redis:7-alpine` with a ping healthcheck
- `pgvector/pgvector:pg16` for Postgres
- `langgraph-api` with `NATS_ENABLED=true` and `NATS_URL=nats://nats:4222`
- `slack-publisher` consuming `agent.review.required`
- `slack-gateway` publishing Slack clicks to `agent.review.feedback`
- `workflow-resumer` consuming feedback and resuming LangGraph
- `otel-collector` receiving OTLP from the gateway and NATS workers
- `jaeger` as the local trace UI the collector forwards to
- optional `mlflow` tracking server behind the `mlflow` Compose profile (`docker compose --profile mlflow up`)
- optional `slack-emulator` behind the `slack-emulator` profile (`docker compose --profile slack-emulator up`)

Do not hard-code database passwords, broker URLs, or collector endpoints in the file; interpolate from `.env`. The obsolete Compose `version:` key can be omitted.

---

## 12. Infrastructure telemetry with OpenTelemetry

Langfuse (next section) answers “what did the agents do?” OpenTelemetry answers “what did the *processes around* the graph do?” Those are different questions. The Slack webhook can return 401, JetStream can redeliver, `workflow_resumer` can fail to reach LangGraph Server — none of that appears in an LLM trace. The FastAPI gateway and the two NATS workers therefore emit their own spans.

[OpenTelemetry](https://opentelemetry.io) is a vendor-neutral standard: a process creates spans, exports them with OTLP, and a **collector** receives that traffic and forwards it to a backend (Jaeger locally, Grafana Cloud, Honeycomb, or another collector). The SDK and the collector are the same whether the process runs in Compose or on a VM in another region.

That split is why every address and token is an environment variable. A `slack-publisher` replica on a laptop and one on a cloud VM use the same code. Only the env changes:

```bash
# Process on the Compose network
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
OTEL_SERVICE_NAME=slack-publisher

# Same binary on a remote host, talking to a collector you already run
OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.prod.example.com:4318
OTEL_EXPORTER_OTLP_HEADERS=authorization=Bearer ${OTEL_TOKEN}
OTEL_SERVICE_NAME=slack-publisher
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=prod,service.namespace=agent-team
```

`configure_otel()` is a no-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is empty or `OTEL_SDK_DISABLED=true`, so tests and a terminal CLI do not need a collector. When the endpoint is set it installs a `BatchSpanProcessor` and an OTLP exporter (HTTP/protobuf or gRPC from `OTEL_EXPORTER_OTLP_PROTOCOL`). The FastAPI app is instrumented except `/health`. JetStream `publish` / `consume` open producer and consumer spans (`messaging.system=nats`). The Slack post and the LangGraph resume are child spans of the consume.

```python
configure_otel(default_service_name="slack-gateway", settings=settings)
# ...
instrument_fastapi(app)

with start_span("nats.publish", kind=SpanKind.PRODUCER, attributes={
    "messaging.system": "nats",
    "messaging.destination.name": subject,
}):
    await js.publish(subject, payload)
```

Client variables:

| Variable | Role |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector or vendor base URL. Required to enable export. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` (default) or `grpc`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth for a remote intake, `key=value,key=value`. |
| `OTEL_SERVICE_NAME` | Process identity. Compose sets a distinct name per worker. |
| `OTEL_RESOURCE_ATTRIBUTES` | Extra resource tags (`deployment.environment`, region, …). |
| `OTEL_SDK_DISABLED` | `true` turns the SDK off even if an endpoint is set. |

Collector variables (the Compose service, or the same collector binary on a server):

| Variable | Role |
| --- | --- |
| `OTEL_COLLECTOR_FORWARD_ENDPOINT` | Next hop. Default `jaeger:4317` on the Compose network. |
| `OTEL_COLLECTOR_FORWARD_INSECURE` | `true` for local Jaeger (no TLS). |
| `OTEL_COLLECTOR_FORWARD_AUTH` | `Authorization` header when the next hop requires a token. |
| `OTEL_COLLECTOR_OTLP_HTTP_PORT` / `_GRPC_PORT` | Host ports mapped to the collector. |

A worker that can reach a vendor OTLP intake does not have to go through the local collector: point `OTEL_EXPORTER_OTLP_ENDPOINT` at the vendor and leave the Compose collector for the processes that stay on that network. The collector is a hop, not a hard dependency of the application code.

Compose interpolates `OTEL_COMPOSE_OTLP_ENDPOINT`, `NATS_COMPOSE_URL`, and `LANGGRAPH_COMPOSE_URL` (docker DNS names) so a host-oriented `.env` (`localhost` NATS, `localhost` collector) does not break containers. A process started *outside* Compose reads `OTEL_EXPORTER_OTLP_ENDPOINT`, `NATS_URL`, and `LANGGRAPH_SERVER_URL` instead.

Local UI: Jaeger on `http://localhost:16686` (override `JAEGER_UI_PORT`). The collector receives OTLP from the apps and forwards to Jaeger. Swap `OTEL_COLLECTOR_FORWARD_ENDPOINT` for any other OTLP backend without touching the workers.

---

## 13. Observability: Langfuse first, MLflow or LangSmith when you need them

LangGraph pairs naturally with LangSmith, but that is not why this architecture traces to Langfuse by default.

LangSmith is proprietary. Cloud SaaS is the default product; self-hosting is an Enterprise contract. For a team that already runs Postgres, Redis, and NATS next to LangGraph Server, that means production traces — prompts, tool calls, HITL pauses — leave the network unless we pay for a plan we do not need. Langfuse is MIT-licensed, self-hostable on every tier, framework-neutral via OpenTelemetry, and priced by volume with no seat fees. That is the better default for LLM-ops: sessions, scores, and prompt versions next to the graph.

`agent_team.tracing` encodes a single active backend so one run is not exported twice:

1. If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, use Langfuse and turn LangSmith tracing off.
2. Else if `MLFLOW_TRACKING_URI` is set, use MLflow (and still keep LangSmith off).
3. Else if `LANGSMITH_TRACING` / `LANGCHAIN_TRACING_V2` and an API key are set, use LangSmith. That keeps the native LangChain pairing for teams that already have a Smith project.
4. Else emit no traces.

Set `LANGFUSE_ENABLED=false` to skip Langfuse even when keys are present, or `MLFLOW_ENABLED=false` to skip MLflow even when a URI is present.

Langfuse integration is the official LangGraph pattern, not a custom exporter bolted onto each node. `configure_tracing()` constructs the Langfuse client; the SDK registers the OpenTelemetry `BatchSpanProcessor` that exports spans to Langfuse. CLI and eval invokes pass `CallbackHandler` through `merge_invoke_config`. The object exported in `langgraph.json` is `build_graph().with_config({"callbacks": [...]})` so LangGraph Server — which never sees the CLI helper — still records every run.

```bash
# Default: self-hosted or cloud Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000   # or https://cloud.langfuse.com
```

Self-host Langfuse with its official production compose (Postgres + ClickHouse), not the LangGraph application compose:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose -f docker-compose.prod.yml up -d
```

Set `NEXTAUTH_URL`, `NEXTAUTH_SECRET`, and the database passwords before exposing it past localhost. Point `LANGFUSE_HOST` at that URL.

### MLflow: what it is, when to use it, how to wire it

[MLflow](https://mlflow.org) is an open-source platform (Apache-2.0) for the classical ML lifecycle: experiment tracking, metrics, parameters, artifacts, and a model registry. Databricks hosts a managed version; anyone can run the tracking server themselves. Since MLflow 2.14 it also records **GenAI traces** — nested spans for LangChain / LangGraph nodes, token usage, and (from 3.6) `thread_id` sessions.

That is a different job from Langfuse. Langfuse is an LLM-ops console. MLflow is an ML platform that happens to understand traces. Use MLflow when:

- the team already has a tracking server or a Databricks workspace, and adding a second product is worse than a slightly less specialized UI;
- agent runs must sit next to training experiments, logged metrics, and registered model versions (for example a coder that later promotes a generated pipeline);
- you want a file-backed or Postgres-backed store you already operate, without an LLM-ops SaaS.

Do **not** pick MLflow as the default LLM tracer if you do not already run it. You will recreate prompt management, session review, and scoring that Langfuse already is.

How the code uses it: when `MLFLOW_TRACKING_URI` is set and Langfuse is not selected, `configure_tracing()` calls `mlflow.set_tracking_uri`, `mlflow.set_experiment`, and `mlflow.langchain.autolog(log_traces=True, run_tracer_inline=True)`. Autolog is the documented LangGraph integration; it patches the LangChain callback manager for the whole process, so the CLI and LangGraph Server both emit traces after the graph module is imported. `run_tracer_inline=True` keeps spans nested on async `ainvoke` (the Slack-resume path). Do not also attach `MlflowLangchainTracer` on `with_config` — that doubles every span. Pass `configurable.thread_id` on every invoke; MLflow 3.6+ groups those traces as a session, which is what you want across a HITL pause and resume.

```python
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "agent-team"))
mlflow.langchain.autolog(log_traces=True, run_tracer_inline=True)

graph.invoke(
    {"messages": [HumanMessage(content=prompt)]},
    {"configurable": {"thread_id": thread_id}},
)
```

Environment variables:

| Variable | Role |
| --- | --- |
| `MLFLOW_TRACKING_URI` | Tracking server or store. Required to select MLflow. Examples: `http://localhost:5000`, `http://mlflow:5000` inside Compose, `file:./mlruns` with no server. |
| `MLFLOW_EXPERIMENT_NAME` | Experiment that owns the traces (default `agent-team`). |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Basic auth against a locked-down server. |
| `MLFLOW_TRACKING_TOKEN` | Bearer token (Databricks and some self-hosted setups). |
| `MLFLOW_ENABLED` | Set `false` to ignore a leftover URI. |

A tracking server is optional for local work (`file:./mlruns` writes under the process working directory). For a UI and a shared store, run the official image. The application Compose file gates that service on the `mlflow` profile so it does not start unless you ask:

```yaml
mlflow:
  profiles: ["mlflow"]
  image: ghcr.io/mlflow/mlflow:v3.3.2
  command:
    - mlflow
    - server
    - --host
    - "0.0.0.0"
    - --port
    - "5000"
    - --backend-store-uri
    - sqlite:////mlflow/mlflow.db
    - --default-artifact-root
    - /mlflow/artifacts
  ports:
    - "5000:5000"
  volumes:
    - mlflow-data:/mlflow
```

```bash
# Host CLI talking to the profile server
export LANGFUSE_ENABLED=false
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=agent-team
docker compose --profile mlflow up -d mlflow
python -m agent_team.cli "Write a Fibonacci function."
# UI: http://localhost:5000
```

From LangGraph Server inside the same Compose network, set `MLFLOW_TRACKING_URI=http://mlflow:5000`. Without Docker:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

SQLite plus a volume is enough to evaluate the backend. Point `--backend-store-uri` at Postgres and `--default-artifact-root` at S3 when you already operate those.

```bash
# Alternative: LangSmith, only when Langfuse keys and MLFLOW_TRACKING_URI are unset
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=multi-agent-slack-pipeline
```

(The older `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` names still work for the LangSmith path.)

A useful eval suite scores three axes:

1. **Code quality** — an LLM judge with structured output (`score` 0–1 plus a comment) against dataset `expected` snippets.
2. **Routing efficiency** — count specialist child runs; three steps is ideal, extra loops decay the score.
3. **Cost and latency** — wall time versus a 45s budget, tokens versus a 15k budget.

The target function must **auto-approve** human review, or the experiment will wait forever on a breakpoint. Pass `configurable.auto_approve=True` (or run the local graph with that config). Seed a LangSmith dataset if it does not exist, then:

```bash
export USE_LOCAL_GRAPH=1
python evals/evaluate_agents.py
```

Use `from langsmith import evaluate` (LangSmith 0.3+). Keep evals out of the request path; they are a CI / regression tool. The harness still talks to LangSmith even when production traces go to Langfuse — evaluations are the remaining LangSmith-shaped job, not the live tracer.

---

## 14. End-to-end flow

1. A client starts a run on assistant `agent_team` with `{"messages": [{"role": "user", "content": "..."}]}` and a `thread_id`.
2. The supervisor sends work to researcher → coder → editor, looping until the artifact looks ready (or `MAX_SUPERVISOR_STEPS` is hit).
3. `review_publisher` writes a `HumanReviewRequested` event to NATS. The graph checkpoints and stops.
4. `slack_publisher` consumes the event and posts the Block Kit card.
5. A reviewer clicks Approve or Reject (optionally with comments).
6. The gateway verifies the signature and publishes `HumanReviewSubmitted` to `agent.review.feedback`.
7. `workflow_resumer` injects `feedback` into the paused thread and creates a new run.
8. `human_review` runs. Approval ends the graph; rejection returns to the supervisor with the new feedback in state.
9. Langfuse (or MLflow, or LangSmith, if that alternative is configured) records each node and token usage. The LangSmith harness, if you ran it, stores experiment scores. The Slack gateway and NATS workers emit OpenTelemetry spans to whatever `OTEL_EXPORTER_OTLP_ENDPOINT` points at.

The same loop works from the terminal: `python -m agent_team.cli "Write a Fibonacci function."`

---

## 15. Pitfalls worth avoiding

**Skipping the review node on resume.** `update_state(..., as_node="human_review")` looks like the docs’ “tell the graph who just ran,” but with `interrupt_before` it bypasses the node that actually interprets feedback.

**Treating Slack `actions` as a dict.** The payload is `actions: [{action_id, value, ...}]`. Index the first element.

**Compiling with `MemorySaver` inside the Server export.** The Platform provides the checkpointer. Export `graph = build_graph()` with `checkpointer=None`.

**Forgetting Redis.** A Postgres-only compose will not run LangGraph Server.

**Calling Slack from the graph.** Publish to JetStream and let `slack_publisher` own retries, credentials, and Block Kit. The graph should still pause if NATS is disabled (CLI), and fail the node if NATS is enabled but unreachable.

**Calling LangGraph from the Slack webhook.** Publish `HumanReviewSubmitted` and let `workflow_resumer` own SDK retries. The gateway only needs to enqueue within Slack’s 3-second window.

**Infinite supervisor loops.** Cap `step_count` and set `recursion_limit` on invoke.

**Using the last chat message as the user request.** After editor turns, that message is a review. Persist `editor_notes` and recover the original human request explicitly.

**Blocking Slack’s 3-second window.** Publish to NATS in the request, then return. Do not wait for LangGraph to resume.

**Evals that hit HITL.** Always auto-approve in the evaluation target.

**Deprecated `config_schema`.** LangGraph 1.x prefers not declaring it; `thread_id` and `auto_approve` still travel on `config["configurable"]`.

---

## 16. What to implement first

A working sequence that avoids boiling the ocean:

1. State, three specialists, supervisor, in-memory checkpointer, CLI interrupt.
2. `interrupt_before`, NATS `HumanReviewRequested`, `slack_publisher` queue worker.
3. Slack gateway publishes `HumanReviewSubmitted`; `workflow_resumer` queue worker resumes LangGraph.
4. Docker Compose with Postgres, Redis, and NATS JetStream.
5. Langfuse tracing (MLflow or LangSmith only if those env vars are set) and the three evaluators.
6. OpenTelemetry on the Slack gateway and NATS workers, with collector URLs only in env vars.

At that point you have a stateful multi-agent pipeline that can research, code, review, pause for a human in Slack, and resume with durable checkpoints—without pretending a linear chain can do the same job.
