# chora-qgen

## About

chora-qgen is a Go service that exposes three subscriber-only agent binaries for question generation, qualitative critique, and generative scene-image rendering. The binaries consume NATS JetStream dispatch requests, call chora-model-gateway for model inference, and use in-memory per-dispatch sessions. The renderer also writes generated images to an S3-compatible object store.

The three binaries are:

| Binary | Dispatch role | Function |
|---|---|---|
| `qgen_question` | `qgen_generate` | Generates or fills one MCQ/OE candidate. `mode=compose` also composes a test-set proposal from accepted candidates and optional source files. |
| `qgen_critic` | `qgen_critique` | Critiques a candidate and returns an accept/reject decision, critique notes, and suggested revisions. |
| `qgen_renderer` | `qgen_render` | Generates a scene image from a render prompt and writes it to the configured render bucket. It can also use a current image as the source for an edit. |

## Quick start

Prerequisites:

- Go 1.26.6, as declared by `go.mod`.
- A NATS JetStream server.
- Access to chora-model-gateway with a tenant ID and GCID.
- For `qgen_renderer`, an S3-compatible object store and a render bucket. MinIO can be used locally.

Clone and build all three binaries:

```bash
git clone https://github.com/apollo-chora/chora-qgen.git
cd chora-qgen

go mod download
go build ./cmd/qgen_question
go build ./cmd/qgen_critic
go build ./cmd/qgen_renderer
```

For a local process, set the required environment and start the desired binary. All binaries are subscriber-only and require dispatch to be enabled:

```bash
export CHORA_PROJECT_ID=your-project
export CHORA_GATEWAY_TENANT_ID=your-tenant
export CHORA_GATEWAY_GCID=your-gcid
export NATS_URL=nats://localhost:4222
export AGENT_DISPATCH_ENABLED=true

./qgen_question
```

Set `QGEN_RENDER_BUCKET` and the S3 variables before starting `qgen_renderer`:

```bash
export QGEN_RENDER_BUCKET=chora-qgen-renders
export S3_ENDPOINT=http://localhost:9000
export S3_ACCESS_KEY_ID=minioadmin
export S3_SECRET_ACCESS_KEY=minioadmin
export S3_REGION=us-east-1

./qgen_renderer
```

## Usage

The binaries do not expose a command-line interface or HTTP API for application work. They subscribe to NATS JetStream dispatch lanes:

| Binary | Request subject |
|---|---|
| `qgen_question` | `chora-qgen-question.agent-dispatch-qgen-generate-requested` |
| `qgen_critic` | `chora-qgen-critic.agent-dispatch-qgen-critique-requested` |
| `qgen_renderer` | `chora-qgen-renderer.agent-dispatch-qgen-render-requested` |

The subscription can be overridden with `AGENT_DISPATCH_SUBSCRIPTION`. The dispatch role is fixed by the binary: `qgen_generate`, `qgen_critique`, or `qgen_render`.

Health checks are served on `AGENT_HEALTH_PORT`, which defaults to `8080`. The available endpoints are:

- `/healthz`
- `/readyz`

### Configuration

Common environment variables:

| Variable | Required | Default / notes |
|---|---|---|
| `CHORA_PROJECT_ID` | yes | Project identifier used in boot logging. |
| `CHORA_AGENT_APP_NAME` | no | ADK session application name. |
| `CHORA_GATEWAY_ENDPOINT` | no | `gateway.chora.site:443`. |
| `CHORA_GATEWAY_AUDIENCE` | no | `https://gateway.chora.site`. |
| `CHORA_GATEWAY_TENANT_ID` | yes | Gateway tenant identity. |
| `CHORA_GATEWAY_GCID` | yes | Gateway GCID. |
| `CHORA_ENV` | no | `dev`, `staging`, or `prod`. |
| `AGENT_DISPATCH_ENABLED` | yes | Must be `true`. |
| `AGENT_DISPATCH_SUBSCRIPTION` | no | Overrides the derived request subscription. |
| `NATS_URL` | yes | NATS JetStream connection URL. |
| `AGENT_HEALTH_PORT` | no | `8080`. |

Model configuration is embedded in `internal/agentconfig/*.yaml` and can be overridden per primary model with these variables:

| Variable | Binary / mode |
|---|---|
| `QGEN_QUESTION_GENERATION_MODEL` | `qgen_question` generation |
| `QGEN_QUESTION_COMPOSE_MODEL` | `qgen_question` `mode=compose` |
| `QGEN_CRITIC_MODEL` | `qgen_critic` |
| `QGEN_RENDERER_MODEL` | `qgen_renderer` |

The embedded fallback chains are not overridden by these environment variables.

For `qgen_renderer`:

| Variable | Required | Default |
|---|---|---|
| `QGEN_RENDER_BUCKET` | yes | none |
| `QGEN_RENDER_TIMEOUT_SECONDS` | no | `120` |
| `QGEN_RENDER_RETRY_ATTEMPTS` | no | `4` |
| `QGEN_RENDER_RETRY_MAX_DELAY_SECONDS` | no | `30` |
| `S3_ENDPOINT` | yes for the renderer's object store | configured by the environment |
| `S3_ACCESS_KEY_ID` | yes for the renderer's object store | configured by the environment |
| `S3_SECRET_ACCESS_KEY` | yes for the renderer's object store | configured by the environment |
| `S3_REGION` | yes for the renderer's object store | configured by the environment |

### Dispatch payloads

`qgen_question` accepts the normal generation path and `mode=compose`.

For generation, the per-turn instructions are selected from `intent` and `question_type`. The implemented question types are MCQ and OE, including fill flows for existing author content.

For `mode=compose`, the dispatch state includes accepted candidate objects with `draft_id`, an `author_prompt`, optional `source_files`, and optional `grounding_mode`. Source files use `gs://` references, and a file with `role="rubric"` is treated as the mark scheme. The completion payload has the form:

```json
{
  "proposed_test_set": {
    "title": "...",
    "description": "...",
    "order": ["draft-id-1"],
    "points": {
      "draft-id-1": 10
    }
  }
}
```

`qgen_critic` reads one candidate from session state and returns:

```json
{
  "accepted": true,
  "critique_notes": "...",
  "suggested_revisions": ["..."]
}
```

Set-style critique responses use a `verdicts` array.

`qgen_renderer` expects a `render_prompt`. Optional fields include `source_image_uri`, `source_image_mime`, `job_id`, and `chunk_id`. Source images must be referenced with `s3://`; edit reads are restricted to the configured render bucket and the dispatch tenant's `tenants/{tenant}/` prefix.

Rendered objects are written as:

```
tenants/{tenant}/jobs/{job-or-execution}/{uuid}.{ext}
```

The renderer completion payload is:

```json
{
  "image_uri": "s3://...",
  "mime_type": "image/png",
  "model_id": "...",
  "model_version": "...",
  "prompt_version": "v1",
  "job_id": "...",
  "chunk_id": "...",
  "bytes": 1234,
  "edit": false
}
```

All three binaries reject command-line arguments and fail to start unless `AGENT_DISPATCH_ENABLED=true`.

## Development

The project is a Go module:

```text
cmd/qgen_question/       qgen_question binary
cmd/qgen_critic/         qgen_critic binary
cmd/qgen_renderer/       qgen_renderer binary
internal/agent/          Prompt composition and session-state handling
internal/agentconfig/    Embedded per-agent model and prompt YAML
internal/boot/           Configuration, agent construction, dispatch, health, and renderer wiring
internal/tools/          ADK tools, including the parse_document tool
.github/workflows/ci.yml CI checks
```

Run the same checks used by CI:

```bash
gofmt -l .
go mod tidy
go vet ./...
go test ./...
```

CI also verifies that `go mod tidy` leaves `go.mod` and `go.sum` unchanged.

Build the Docker image with the repository's Dockerfile:

```bash
docker buildx build --platform=linux/amd64 \
  --build-arg SERVICE_NAME=chora-qgen \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t walfa/chora-qgen:latest .
```

The image contains all three binaries and uses `/app/qgen_question` as its default entrypoint. The container command can select `/app/qgen_critic` or `/app/qgen_renderer`.

The `internal/tools/parse_document` ADK tool is currently a fail-loud stub. It returns `doc_parser_not_yet_wired` rather than fabricated document content.
