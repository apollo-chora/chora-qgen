// Command qgen_critic is the entry point for the 2-agent qgen crew's
// critic agent — the second of two members (qgen_question generates,
// qgen_critic critiques).
//
// User-locked 2026-05-17 (per docs/m13/ack-oe-ai-assist-plan-2026-05-17.md):
//
//   - DISTINCT crew from qgen_question (3-agent ADR-153 pipeline) and from
//     the 6-agent ai_assist_crew legacy content gate.
//   - "Critic" role NOT "evaluator" — qualitative critique, NOT scoring.
//     The 6-agent crew's `evaluator` is a different concern (scoring +
//     content-gate); a separate scoring agent is on the roadmap.
//   - LangGraph orchestrator owns the quality loop: generate → critique →
//     quality_gate (≤max_retries) → publish completed.v1 (with
//     quality_warning if retries exhausted) OR refused.v1 (guardrail).
//
// Pipeline shape (single agent — no internal sub-pipeline):
//
//  1. qgen_critic                            (T1 LLM-as-critic)
//     Tool-free; reads one AiAssistCandidate JSON; emits one CritiqueOutput
//     JSON: {accepted, critique_notes, suggested_revisions}.
//
// Per-QuestionType prompt branching happens inside ComposeCriticInstruction
// (composer is a pure function per ADR-141 D2 transparency). MCQ + OE
// rubrics share the [CONTEXT]/[ROLE]/[EXAMPLES]/[AUDIENCE] frame; the
// [TASK] + [EXPECTED OUTPUT] blocks branch.
//
// Per user direction 2026-05-17 (this session):
//   - Deploy via adkgo (NOT gcloud agents deploy; that command doesn't
//     exist) per [[agents-cli-cicd]] skill
//   - Register in chora-infra/agents-cli/registry.json after deploy
//   - OTel traces emit via OTLP per [[ai-observability]] (ADK Go SDK
//     auto-instruments; the plugin chain adds Chora-specific span
//     attributes)
//   - D6 4-pillar resilience per [[agentic-resilience-d6]] — pod-death
//     survival via the managed runtime; idempotent semantics (critic is a
//     pure function of input candidate); DLQ owned by the orchestrator's
//     outbox pattern
//
// Env vars (NEVER inlined per feedback_no_inline_config):
//
//	CHORA_PROJECT_ID                       — required; stamped on the boot log line
//	CHORA_AGENT_APP_NAME                   — ADK session AppName label (optional)
//	QGEN_CRITIC_MODEL                      — primary model override (default: embedded agentconfig YAML)
//	CHORA_GATEWAY_ENDPOINT                 — default gateway.chora.site:443
//	CHORA_GATEWAY_TENANT_ID / _GCID        — required (ADR-163; per-request values from the dispatch win)
//	CHORA_ENV                              — dev | staging | prod
//	AGENT_DISPATCH_ENABLED                 — must be "true" (no other transport)
//	AGENT_DISPATCH_SUBSCRIPTION            — request subscription (chora-qgen-critic.agent-dispatch-qgen-critique-requested)
//	NATS_URL                               — NATS JetStream event bus of the dispatch lanes
//	AGENT_HEALTH_PORT                      — health port (default 8080)
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
	// SIGTERM (rollout, scale-down) cancels the context: the dispatch
	// subscriber stops receiving, in-flight work finishes, the process exits
	// 0; anything else is an error and exits non-zero (ADR-254 D6). The binary
	// refuses any argument: the web launcher is gone.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	if err := boot.RunCritic(ctx, os.Args[1:]); err != nil {
		// slog.Error + os.Exit(1), NOT log.Fatalf: slog.SetDefault routes
		// the standard log package through this handler at INFO, which
		// would emit every boot failure below the level any log-based
		// alert watches for. The boot errors already name the binary.
		slog.Error("qgen_critic boot failed", "err", err)
		os.Exit(1)
	}
}
