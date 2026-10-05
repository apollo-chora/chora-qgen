# chora-qgen

The QGen ADK Go agent crew — three subscriber-only binaries that back
chora-creation's AI-assist question authoring and the generative scene-image
render path:

| Binary | Dispatch role | Purpose |
|---|---|---|
| `cmd/qgen_question` | `qgen_generate` | Single-question AI-assist generation (ADR-153): drafts or fills one MCQ/OE candidate per author request, plus the `mode=compose` testset composer. |
| `cmd/qgen_critic` | `qgen_critique` | Qualitative critique of one candidate (or a set) for the orchestrator's quality loop — accept/reject + revision notes, never scoring. |
| `cmd/qgen_renderer` | `qgen_render` | The generative scene image per chunk: calls the model gateway with `response_modality=IMAGE` and writes the image to the render bucket. |

Module path: `github.com/apollo-chora/chora-qgen`.

The crew is cloud-neutral: NATS JetStream for events (via
`chora-adk-common/agentdispatch`), standard OTLP for traces (via
`chora-adk-common/tracing`), env-backed secrets, an S3-compatible object store
for rendered images (via `chora-common/objectstore`), and the model-gateway gRPC
adapter for model calls (via `chora-adk-common/modelgatewayclient`). No cloud
account or managed service is required.

## Behaviour

- **qgen_question** serves the `qgen_generate` dispatch lane. The per-turn
  system prompt is re-composed from session state (`intent`, `question_type`,
  `author_*`) on every turn; the generation step branches over 4 prompt
  templates (new_mcq / new_oe / fill_mcq / fill_oe). `mode=compose` routes to
  the testset composer (composer LLM + deterministic finaliser); any other mode
  is a permanent `unknown_mode` before any model call.
- **qgen_critic** serves the `qgen_critique` lane: a tool-free single-shot agent
  that reads the candidate JSON from session state and emits
  `{accepted, critique_notes, suggested_revisions}` (single) or
  `{verdicts: [...]}` with a strict candidate_id echo contract (set mode).
- **qgen_renderer** serves the `qgen_render` lane: parses the scene prompt,
  optionally reads the current image of an edit by `s3://` reference (refused
  permanently unless it is inside the render bucket under the dispatch
  tenant's own prefix), invokes the image model, writes
  `tenants/{tenant}/jobs/{job|execution}/{uuid}.{ext}` and answers the
  completion envelope `{image_uri, mime_type, model_id, model_version,
  prompt_version, job_id, chunk_id, bytes, edit}`.

All three binaries take no arguments, run stateless per-dispatch sessions, and
refuse to start unless `AGENT_DISPATCH_ENABLED=true` (the dispatch subscriber is
their only transport).

## Layout

| Path | Purpose |
|---|---|
| `cmd/qgen_question/` | The generator binary (no arguments). |
| `cmd/qgen_critic/` | The critic binary (no arguments). |
| `cmd/qgen_renderer/` | The renderer binary (no arguments). |
| `internal/boot/` | Boot wiring: env + embedded agentconfig resolution, agent tree, plugin chain, subscriber serve, render envelope. |
| `internal/agent/` | Pure prompt composers (question / critic / compose) and session-state readers. |
| `internal/agentconfig/` | Embedded per-agent model + prompt YAML (single source of truth for tier + fallback chain). |
| `internal/tools/` | The `parse_document` ADK tool (fail-loud stub until chora-doc-parser lands). |

## Transport and dependencies

- **Events** — NATS JetStream via `chora-common/eventbus` (inside
  `chora-adk-common/agentdispatch`). The request subscriptions
  (`chora-qgen-{question,critic,renderer}.agent-dispatch-qgen-{generate,critique,render}-requested`)
  are valid NATS subjects; the canonical event envelope rides as NATS headers.
- **Traces** — standard OTLP/gRPC via `chora-common/otel`; stdout in local dev
  when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset.
- **Model calls** — gRPC to `chora-model-gateway`; TLS + `CHORA_GATEWAY_TOKEN`
  in production, plaintext when `CHORA_GATEWAY_INSECURE` is set for local dev.
- **Object store** — S3-compatible (MinIO locally) via `chora-common/objectstore`
  for the renderer's image bucket.

Each binary listens on the health port `AGENT_HEALTH_PORT` (default `8080`,
`/healthz` + `/readyz`) and needs NATS plus the model gateway. The renderer
additionally needs the object store (`S3_ENDPOINT` / `S3_ACCESS_KEY_ID` /
`S3_SECRET_ACCESS_KEY` / `S3_REGION`) and `QGEN_RENDER_BUCKET`. No database.

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `CHORA_PROJECT_ID` | yes | Stamped on the boot log line. |
| `CHORA_AGENT_APP_NAME` | no | ADK session AppName label. |
| `CHORA_GATEWAY_ENDPOINT` | no | Default `gateway.chora.site:443`. |
| `CHORA_GATEWAY_TENANT_ID` / `CHORA_GATEWAY_GCID` | yes | Gateway identity (ADR-163). |
| `CHORA_GATEWAY_AUDIENCE` | no | Default `https://gateway.chora.site`. |
| `CHORA_ENV` | no | `dev` \| `staging` \| `prod`. |
| `AGENT_DISPATCH_ENABLED` | yes | Must be `true` (subscriber-only). |
| `AGENT_DISPATCH_SUBSCRIPTION` | no | Override the derived request subscription. |
| `NATS_URL` | yes | NATS JetStream event bus. |
| `AGENT_HEALTH_PORT` | no | Default `8080`. |
| `QGEN_QUESTION_GENERATION_MODEL` / `QGEN_QUESTION_COMPOSE_MODEL` / `QGEN_CRITIC_MODEL` / `QGEN_RENDERER_MODEL` | no | Primary model overrides (default: embedded agentconfig YAML). |
| `QGEN_RENDER_BUCKET` | renderer | The render bucket. |
| `QGEN_RENDER_TIMEOUT_SECONDS` / `QGEN_RENDER_RETRY_ATTEMPTS` / `QGEN_RENDER_RETRY_MAX_DELAY_SECONDS` | no | Image invoke timeout + retry budget. |

## Docker

```bash
docker buildx build --platform=linux/amd64 \
  --build-arg SERVICE_NAME=chora-qgen \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t walfa/chora-qgen:latest .
```

One image, three binaries; the container `command` selects the member
(default `/app/qgen_question`).
