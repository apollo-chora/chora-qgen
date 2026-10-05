// Command qgen_renderer is the third member of the qgen crew (ADR-254 D2 /
// D12): the generative scene image per chunk, served SUBSCRIBER-ONLY on the
// dispatch role qgen_render. The kennel's render_chunk node keeps the
// deterministic Kroki render and the signed-URL step; this binary takes the
// composed scene prompt (and, for an ADR-210 edit, the current image by s3://
// reference), calls chora-model-gateway with response_modality=IMAGE
// (surface qgen, agent_id qgen_renderer, the dispatch key stamped, un-metered
// per D7), writes the image to the render bucket and returns its s3:// URI.
// No ADK web launcher, a health port (/healthz, /readyz) and nothing else; a
// subscriber that cannot start ends the process non-zero; no arguments.
//
// Env vars (NEVER inlined per feedback_no_inline_config):
//
//	CHORA_PROJECT_ID                       — required; stamped on the boot log line
//	CHORA_AGENT_APP_NAME                   — ADK session AppName label (optional)
//	CHORA_GATEWAY_ENDPOINT                 — default gateway.chora.site:443
//	CHORA_GATEWAY_TENANT_ID / _GCID        — required (ADR-163; per-request values from the dispatch win)
//	QGEN_RENDER_BUCKET                     — required: the render bucket (the kennel signs reads from it)
//	QGEN_RENDERER_MODEL                    — image model override (default: embedded agentconfig YAML)
//	QGEN_RENDER_TIMEOUT_SECONDS            — per-attempt Invoke ceiling (default 120)
//	QGEN_RENDER_RETRY_ATTEMPTS             — retry budget on a throttled model (default 4)
//	QGEN_RENDER_RETRY_MAX_DELAY_SECONDS    — backoff cap (default 30)
//	CHORA_ENV                              — dev | staging | prod
//	AGENT_DISPATCH_ENABLED                 — must be "true" (no other transport)
//	AGENT_DISPATCH_SUBSCRIPTION            — request subscription (chora-qgen-renderer.agent-dispatch-qgen-render-requested)
//	NATS_URL                               — NATS JetStream event bus of the dispatch lanes
//	AGENT_HEALTH_PORT                      — health port (default 8080)
//	S3_ENDPOINT / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_REGION
//	                                     — the object store the render bucket lives in (MinIO locally)
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/apollo-chora/chora-qgen/internal/boot"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	if err := boot.RunRenderer(ctx, os.Args[1:]); err != nil {
		slog.Error("qgen_renderer: fatal", "err", err.Error())
		os.Exit(1)
	}
}
