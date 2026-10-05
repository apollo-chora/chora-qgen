// Command qgen_question is the entry point for the 3-agent qgen_question
// crew per ADR-153 — single-question AI-assist surface for chora-creation
// Question Authoring CR (ai_draft + ai_model_answer paths).
//
// Per ADR-153 + crew-composition SKILL §3 the pipeline is:
//
//  1. content_assurance ⇄ doc_parser_tool   (T1, multimodal-capable)
//     Registers parse_document tool (M14.0 fail-loud stub; M14.1 wires
//     chora-doc-parser per qgen-crew-composition §9 item 7)
//  2. qgen_question_generation              (T2)
//     4 prompt templates per Intent × QuestionType (composer-resolved)
//  3. qgen_question_evaluation              (T1 LLM-as-judge)
//     Same 3 axes as the 6-agent crew (factuality / clarity / difficulty)
//
// Each step is its own `llmagent.New(...)` with its OWN modelgatewayclient
// LLM (Phase 3.1 cutover per ADR-163; was direct gemini.NewModel until
// 2026-05-25). The gateway preserves per-agent logical_model_id routing
// so the slim-crew tiering shape is unchanged — only the chokepoint
// moved. The pipeline is wrapped by
// `sequentialagent.New(...)`.
//
// M14.2 — per-turn instruction provider (this commit):
//
//	Every sub-agent's system prompt is re-composed PER TURN from
//	session.State() via stepInstructionProvider. Reads `intent`,
//	`question_type`, and (for IntentModelAnswerFill) `author_stem` /
//	`author_options` / `author_rubric` / `model_answer` — keys the
//	Python executor writes in _build_session_state. Replaces the prior
//	boot-time static compose that hard-coded
//	{Intent: new_question, QuestionType: mcq} for every request and
//	caused the OE smoke to fail (live engine emitted MCQ regardless of
//	state). See BuildTaskContextFromState in internal/agent/composer_question.go.
//
// Subscriber-only deploy: the binary serves ONLY the NATS JetStream dispatch
// lane (AGENT_DISPATCH_ENABLED=true); there is no web launcher and no hosted
// runtime. The kennel's executor stamps the per-turn session state; the
// completion envelope rides the reply subject.
//
// Env vars (NEVER inlined per feedback_no_inline_config):
//
//	CHORA_PROJECT_ID                       — required; stamped on the boot log line
//	CHORA_AGENT_APP_NAME                   — ADK session AppName label (optional)
//	QGEN_QUESTION_GENERATION_MODEL         — primary model override (default: embedded agentconfig YAML)
//	QGEN_QUESTION_COMPOSE_MODEL            — compose model override (default: embedded agentconfig YAML)
//	CHORA_GATEWAY_ENDPOINT                 — default gateway.chora.site:443
//	CHORA_GATEWAY_TENANT_ID / _GCID        — required (ADR-163; per-request values from the dispatch win)
//	CHORA_ENV                              — dev | staging | prod
//	AGENT_DISPATCH_ENABLED                 — must be "true" (no other transport)
//	AGENT_DISPATCH_SUBSCRIPTION            — request subscription (chora-qgen-question.agent-dispatch-qgen-generate-requested)
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

// sample comment
func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))
	// SIGTERM (rollout, scale-down) cancels the context: the dispatch
	// subscriber stops receiving, in-flight work finishes, the process exits
	// 0; anything else is an error and exits non-zero (ADR-254 D6). The binary
	// refuses any argument: the web launcher is gone.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	if err := boot.RunQuestion(ctx, os.Args[1:]); err != nil {
		// slog.Error + os.Exit(1), NOT log.Fatalf: slog.SetDefault routes
		// the standard log package through this handler at INFO, which
		// would emit every boot failure below the level any log-based
		// alert watches for. The boot errors already name the binary.
		slog.Error("qgen_question boot failed", "err", err)
		os.Exit(1)
	}
}
