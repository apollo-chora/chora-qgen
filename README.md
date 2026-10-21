# chora-qgen

> The question-generation crew: one agent writes questions, one critiques them, one renders
> the illustrations.

Go. Three subscriber binaries, one image, no database. They consume agent-dispatch requests
over NATS JetStream and call the platform's model gateway.

## What it is

The `qgen` crew is the engine behind the authoring canvas. Three roles, each a separate
binary and a separate Compose service:

| Binary | Dispatch role | Service | Job |
| --- | --- | --- | --- |
| `cmd/qgen_question` | `qgen_generate` | `qgen-question` | writes candidate questions, or composes an accepted set |
| `cmd/qgen_critic` | `qgen_critique` | `qgen-critic` | accepts or rejects candidates with notes and suggested revisions |
| `cmd/qgen_renderer` | `qgen_render` | `qgen-renderer` | renders a scene image and writes it to object storage |

None serves application traffic. All three take no command-line arguments, and all three
refuse to start unless `AGENT_DISPATCH_ENABLED=true`.

### The health endpoints work here — unlike their older siblings

All three answer health **for real** in this stack:

```sh
docker compose exec -T nats wget -qO- http://qgen-renderer:8080/readyz   # → ready
```

That is worth calling out because it is not universal across the platform. `chora-companion-diagnosis`,
`chora-kg-explorer` and `chora-oe-grading` log a `health_addr` and even `agentdispatch.serving`
while nothing listens on the port. The difference is the dependency pin:

| Repository | `chora-adk-common` pin | Health listener |
| --- | --- | --- |
| `chora-qgen` | `v0.0.0-20261018001513-336709b355a0` | **binds** |
| `chora-companion-diagnosis`, `chora-kg-explorer`, `chora-oe-grading` | `v0.0.0-20261005110720-56af53fbecf8` | does not bind |

So this is not a framework-wide defect — it is an old pin. The reliable check when in doubt is
still the broker's consumer list, but here `GET /healthz` and `/readyz` are usable.

### Four model variables are mandatory

Unlike most services in the stack, `compose.yml` enforces these at interpolation time with
`${…:?…}`, so **the stack will not run a single Compose command** while any is unset:

```text
QGEN_QUESTION_GENERATION_MODEL   QGEN_QUESTION_COMPOSE_MODEL
QGEN_CRITIC_MODEL                QGEN_RENDERER_MODEL
```

That is why four of the stack's 17 required variables come from this service. They are
**logical model ids the gateway resolves**, not provider model names, and they may all name the
same route. The embedded fallback chains in `internal/agentconfig/*.yaml` are **not** overridable
by environment — only the primary model is.

## Prerequisites

| Route | What you need |
| --- | --- |
| In the stack | A working `chora-stack` deployment. NATS, the model gateway and Garage come from it. |
| Standalone | Go **1.26.6** (`go.mod` declares it), a reachable NATS JetStream broker, and a reachable model gateway. |
| Renderer only | An object store: endpoint, key pair, region, and a bucket. |
| Either | Gateway identity and the four model ids above. |

## Credentials

The crew holds **one** credential plus one storage key pair — no model-provider keys.

| Value | Where it comes from | Variable |
| --- | --- | --- |
| Gateway caller token | the stack's `GATEWAY_API_KEY` | `CHORA_GATEWAY_TOKEN` |
| Gateway tenant / GCID | the stack's `CHORA_DEFAULT_TENANT_ID` / `CHORA_DEFAULT_GCID` | `CHORA_GATEWAY_TENANT_ID`, `CHORA_GATEWAY_GCID` |
| NATS endpoint | the stack's NATS service; no credential locally | `NATS_URL` |
| Four model ids | the stack's `.env`; **all required** | see above |
| Object-store credentials | the stack's Garage key and secret (renderer only) | `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` |

`compose.yml` marks `GATEWAY_API_KEY` required, so the stack will not start the crew without
it. Locally the value is inert — `GATEWAY_AUTH_ENABLED=false` — but it must be non-empty and
must equal the `<key>` half of `GATEWAY_API_KEYS` before gateway authentication is switched on.

The provider key behind the model routes belongs to the gateway, not to these agents. Where it
comes from, and how to obtain it, is documented once in
[`chora-stack/README.md`](../chora-stack/README.md#credentials).

## Run it in the stack

`compose.yml` builds the image from `../chora-qgen` **three times**, overriding the entrypoint
each time with `/app/qgen_question`, `/app/qgen_critic` and `/app/qgen_renderer`.

All three start once `nats` is healthy, `nats-init` has completed, and `model-gateway` has
started. None publishes a host port.

```sh
cd chora-stack
docker compose up -d qgen-question qgen-critic qgen-renderer
docker compose ps qgen-question qgen-critic qgen-renderer
```

What Compose sets per service: the gateway endpoint (`model-gateway:9090`), audience
(`chora-local`), `CHORA_GATEWAY_INSECURE=1` for plaintext gRPC inside the Compose network,
`CHORA_ENV=local`, `AGENT_DISPATCH_ENABLED=true`, `AGENT_HEALTH_PORT=8080`, and each role's
model id.

The renderer additionally gets the storage configuration:

| Variable | Stack value |
| --- | --- |
| `QGEN_RENDER_BUCKET` | `chora-ai-assist-images` |
| `S3_BUCKET` | `chora-ai-assist-images` |
| `S3_ENDPOINT` | `http://garage:3900` |
| `S3_FORCE_PATH_STYLE` | `true` — Garage needs path-style addressing |
| `S3_REGION` | `us-east-1` |

The dispatching half also has to be on: the kernel starts the `qgen` lane only when
`QGEN_CREW_ENABLED` is set, which the stack does. Either half missing means work is published
to a subject nobody consumes, silently.

## Run it standalone

```sh
git clone https://github.com/apollo-chora/chora-qgen.git
cd chora-qgen
go build ./...
```

```sh
export CHORA_PROJECT_ID=local
export CHORA_GATEWAY_ENDPOINT=localhost:9090   # else it defaults to gateway.chora.site:443
export CHORA_GATEWAY_INSECURE=1
export CHORA_GATEWAY_AUDIENCE=chora-local
export CHORA_GATEWAY_TENANT_ID=your-tenant-id
export CHORA_GATEWAY_GCID=your-gcid
export CHORA_GATEWAY_TOKEN=<gateway caller token>
export QGEN_QUESTION_GENERATION_MODEL=<logical model id>
export QGEN_QUESTION_COMPOSE_MODEL=<logical model id>
export QGEN_CRITIC_MODEL=<logical model id>
export QGEN_RENDERER_MODEL=<logical model id>
export NATS_URL=nats://localhost:4222
export AGENT_DISPATCH_ENABLED=true
go run ./cmd/qgen_question
```

Swap the command for `./cmd/qgen_critic` or `./cmd/qgen_renderer`. The binaries do not read
`.env` — export the values. For the renderer, add `QGEN_RENDER_BUCKET` **and all four** `S3_*`
values (`S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION`) — omitting the
region leaves the object-store client incompletely configured. The gateway three-liner matters:
without `CHORA_GATEWAY_ENDPOINT` and `CHORA_GATEWAY_INSECURE` the agent dials
`gateway.chora.site:443` over TLS and will not reach a gateway on your machine.

Container: the image defaults to `/app/qgen_question`; select the others by command:

```sh
docker build -t chora-qgen .
docker run --rm chora-qgen                          # question
docker run --rm chora-qgen /app/qgen_critic         # critic
docker run --rm chora-qgen /app/qgen_renderer       # renderer
```

## Configuration

**Common**

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `CHORA_PROJECT_ID` | yes | — | project identifier in boot logging |
| `CHORA_AGENT_APP_NAME` | no | unset | ADK session app name |
| `CHORA_GATEWAY_ENDPOINT` | no | `gateway.chora.site:443` | gateway gRPC target |
| `CHORA_GATEWAY_AUDIENCE` | no | `https://gateway.chora.site` | token audience |
| `CHORA_GATEWAY_INSECURE` | no | unset | plaintext gRPC when truthy |
| `CHORA_GATEWAY_TENANT_ID` | yes | — | gateway tenant identity |
| `CHORA_GATEWAY_GCID` | yes | — | gateway GCID |
| `CHORA_GATEWAY_TOKEN` | stack | unset in code | gateway caller token |
| `CHORA_ENV` | no | `dev` | environment label |
| `AGENT_DISPATCH_ENABLED` | yes | `false` | must be `true` |
| `AGENT_DISPATCH_SUBSCRIPTION` | no | derived | request-subscription override |
| `NATS_URL` | yes | — | NATS JetStream URL |
| `AGENT_HEALTH_PORT` | no | `8080` | health port — **bound in this service** |

**Models** (primary override; fallback chains are embedded and not overridable)

| Variable | Applies to |
| --- | --- |
| `QGEN_QUESTION_GENERATION_MODEL` | `qgen_question` generation |
| `QGEN_QUESTION_COMPOSE_MODEL` | `qgen_question` `mode=compose` |
| `QGEN_CRITIC_MODEL` | `qgen_critic` |
| `QGEN_RENDERER_MODEL` | `qgen_renderer` |

**Renderer storage**

| Variable | Required | Default |
| --- | --- | --- |
| `QGEN_RENDER_BUCKET` | yes | none |
| `QGEN_RENDER_TIMEOUT_SECONDS` | no | `120` |
| `QGEN_RENDER_RETRY_ATTEMPTS` | no | `4` |
| `QGEN_RENDER_RETRY_MAX_DELAY_SECONDS` | no | `30` |
| `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION` | yes for the renderer | environment |

## Verify it works

### It is up

```sh
for s in qgen-question qgen-critic qgen-renderer; do
  docker compose exec -T nats wget -qO- "http://$s:8080/readyz"; echo
done
```

```text
ready
ready
ready
```

Unlike the other agent services, these endpoints are real — see
[The health endpoints work here](#the-health-endpoints-work-here--unlike-their-older-siblings).

### The broker knows all three consumers

```sh
curl -s 'http://localhost:8222/jsz?consumers=1' \
  | grep -o 'chora-qgen-[a-z]*-agent-dispatch[a-z-]*' | sort -u
```

```text
chora-qgen-critic-agent-dispatch-qgen-critique-requested
chora-qgen-question-agent-dispatch-qgen-generate-requested
chora-qgen-renderer-agent-dispatch-qgen-render-requested
```

### The dispatching lane is on

```sh
docker compose exec -T nats wget -qO- http://ai-kernel-orchestrator:8080/readyz \
  | grep -o '"qgen":{[^}]*}'
```

`"started": true` with all three roles is required. With the crew running and the lane off,
generation requests are published to a subject with no consumer.

### The renderer can write to its bucket

```sh
cd chora-stack && ./scripts/verify-routes.sh
```

The route probes include a real image generation through this stack's model gateway, which is
the same path the renderer uses. A `PASS` there means the model route works; to confirm the
object itself landed, list `chora-ai-assist-images` through Garage's S3 API on
`localhost:9000`, where the key has the form
`tenants/{tenant}/jobs/{job-or-execution}/{uuid}.{ext}`.

## Troubleshooting

- **A Compose command reports a missing `QGEN_*_MODEL`** — one of the four model ids is unset.
  All four are `:?`-required, so this gates *every* command in `chora-stack`, not just this
  service.
- **Boot fails naming a model variable** — same cause, per binary. The fallback chains do not
  rescue an empty primary.
- **The service exits immediately** — `AGENT_DISPATCH_ENABLED` is not `true`, or gateway
  identity is missing.
- **Generation requests are published but nothing consumes them** — either the kernel's
  `QGEN_CREW_ENABLED` is off, or the publisher is using a different subject. Compare with the
  broker consumer list.
- **Generation works but no image appears** — the renderer is a separate role. Check
  `qgen-renderer`'s logs and that the kernel dispatched `qgen_render`; the question and critic
  lanes can be healthy while the render lane is idle.
- **The renderer fails on a `gs://` source image** — only `s3://` is accepted for source
  images, and edit reads are restricted to the configured render bucket under the dispatch
  tenant's `tenants/{tenant}/` prefix.
- **Rendered objects are not where you expect** — the key is
  `tenants/{tenant}/jobs/{scope}/{32-hex}.{ext}`. Three details catch people: `{scope}` falls
  back to the execution id and then to the literal `unscoped` when no `job_id` is supplied; the
  filename is a **UUIDv7 with its hyphens stripped**, so it is 32 hex characters and will not
  match a canonical-UUID search; and the extension comes from the MIME type, defaulting to `png`.
- **An image request returns `finishReason: IMAGE_RECITATION` with no image** — upstream content
  policy, not a route failure. This is raised as a **permanent** `image_blocked` failure, so a
  redelivery will not retry it — dispatch again with a different prompt. Prefer a distinctive,
  specific scene; generic subjects are the usual trigger.
- **A candidate that carries any image spec aborts the whole generation** — the crew refuses a
  job with `image_specs` when no renderer client is configured. In the stack the renderer is
  present, so this points at configuration rather than absence.
- **Traces never reach Tempo** — these agents pass `otel-collector:4317` without an `http://`
  prefix, so they are **not** affected by the `too many colons` error seen elsewhere in the
  stack.

## API surface

No inbound HTTP API beyond health. Each binary consumes one dispatch subject and returns its
result through the same bus.

| Role | Subscription | Consumes | Returns |
| --- | --- | --- | --- |
| `qgen_generate` | `chora-qgen-question.agent-dispatch-qgen-generate-requested` | generation or `mode=compose` state | candidates, or `{"proposed_test_set": {title, description, order, points}}` |
| `qgen_critique` | `chora-qgen-critic.agent-dispatch-qgen-critique-requested` | one candidate | `{"accepted", "critique_notes", "suggested_revisions"}`; set-style uses `verdicts` |
| `qgen_render` | `chora-qgen-renderer.agent-dispatch-qgen-render-requested` | `render_prompt`, optional `source_image_uri`, `source_image_mime`, `job_id`, `chunk_id` | `{image_uri, mime_type, model_id, model_version, prompt_version, job_id, chunk_id, bytes, edit}` |

`qgen_question` supports MCQ and OE, including fill flows for existing author content. In
`mode=compose` the dispatch state carries accepted candidates with `draft_id`, an
`author_prompt`, optional `source_files` and optional `grounding_mode`; a source file with
`role="rubric"` is treated as the mark scheme.

## Architecture

```text
cmd/qgen_question/     generation and compose
cmd/qgen_critic/       accept/reject with notes
cmd/qgen_renderer/     scene image generation and object-store write
internal/agent/        prompt composition and output handling per role
internal/agentconfig/  embedded per-role model, tier and prompt-version YAML
internal/boot/         configuration, gateway wiring, dispatch setup, startup
Dockerfile             one image, three binaries
```

The renderer is the only one with state outside the bus: it writes to object storage and
returns an `s3://` URI rather than image bytes. That keeps large payloads off NATS, and makes
the bucket configuration part of the deployment rather than an implementation detail.

## Events

Dispatch is NATS JetStream, one durable consumer per role, with request subjects derived from
the service and role unless `AGENT_DISPATCH_SUBSCRIPTION` overrides them.

| Concern | Setting |
| --- | --- |
| Streams | `CHORA_EVENTS` for normal dispatch, `CHORA_DLQ` for dead letters |
| Redelivery | Durable consumers; the ceiling comes from `chora-adk-common` |
| Renderer timeout | `QGEN_RENDER_TIMEOUT_SECONDS`, default 120s |
| Renderer retries | `QGEN_RENDER_RETRY_ATTEMPTS`, default 4, with exponential backoff capped by `QGEN_RENDER_RETRY_MAX_DELAY_SECONDS` |

The three roles form a pipeline the kernel drives: generate → critique → (on acceptance)
render. The renderer is dispatched per image spec, so one generation can produce several render
calls.

## Migrations

**None.** None of the three binaries has a database, the repository ships no `migrations/`
directory, and the crew is not part of the stack's migrate one-shot set.

If you are auditing a deployment and expect a `qgen-migrate` container, there isn't one and
there should not be.
