package boot

import (
	"fmt"
	"google.golang.org/genai"
	"iter"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/session"

	"github.com/apollo-chora/chora-adk-common/agentdispatch"
	"github.com/apollo-chora/chora-adk-common/promptstamping"

	qgenagent "github.com/apollo-chora/chora-qgen/internal/agent"
)

// StepInstructionProvider returns an llmagent.InstructionProvider that
// re-composes the [CREATE]-block prompt for the given canonical Step
// PER TURN. Reads session.State(), the same readonly view manaplugin +
// instancedispatch use, and feeds it to BuildTaskContextFromState.
//
// This is the M14.2 fix: prior boot-time compose hard-coded
// TaskContextQuestion{Intent: IntentNewQuestion, QuestionType: QuestionTypeMCQ}
// so the live engine emitted MCQ for every request regardless of state.
// Now the LLM-issuing path re-reads `intent` / `question_type` /
// `author_*` from state on each turn and folds them into the system
// prompt before the model call.
//
// The Python executor populates state via _build_session_state.
// qgen_generate payload discriminator (ADR-254 D6): mode absent or
// "generate" is the question generation this pipeline serves; "compose" (the
// testset composer, ADR-254 D2) is refused by name until the composer port
// lands in this binary, so a compose request can never silently run
// generation; anything else is unknown_mode. All permanent (no redelivery can
// change a payload).
const (
	stateKeyMode = "mode"
	ModeGenerate = "generate"
	ModeCompose  = "compose"
)

// selectMode reads the discriminator before any model call: generate
// (default) or compose; anything else is a permanent unknown_mode.
func selectMode(st interface {
	Get(string) (any, error)
}) (string, error) {
	switch mode := strings.ToLower(strings.TrimSpace(StateString(st, stateKeyMode))); mode {
	case "", ModeGenerate:
		return ModeGenerate, nil
	case ModeCompose:
		return ModeCompose, nil
	default:
		return "", agentdispatch.Permanent("unknown_mode: "+mode, nil)
	}
}

func StepInstructionProvider(step qgenagent.Step) llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		taskCtx, err := qgenagent.BuildTaskContextFromState(ctx.ReadonlyState())
		if err != nil {
			return "", err
		}
		// BatchID is currently not threaded through session state from
		// the executor (chora-creation owns batching upstream). Leaving
		// it empty keeps the [CONTEXT] block's "batch_id=<unset>" cue
		// that is the same render the static boot-time compose used.
		return qgenagent.ComposeQGenQuestionInstruction(step, taskCtx), nil
	}
}

// CriticInstructionProvider returns an llmagent.InstructionProvider that
// re-composes the critic's CREATE prompt PER TURN from session.State(). This is
// the M14.2 fix (2026-06-01, user directive arch-clean/debt-free/fail-loud):
// the prior boot-time static compose hard-coded "<filled at runtime>"
// placeholders and never injected the per-request candidate, so the critic
// always answered "input_unparseable" and forced the orchestrator quality loop
// to max_attempts on every job. BuildCriticTaskContextFromState fails loud
// (returns an error) when the candidate is missing, surfacing the pipeline bug
// instead of silently rejecting an empty candidate. Mirrors
// StepInstructionProvider.
func CriticInstructionProvider(step qgenagent.Step) llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		taskCtx, err := qgenagent.BuildCriticTaskContextFromState(ctx.ReadonlyState())
		if err != nil {
			return "", err
		}
		return qgenagent.ComposeCriticInstruction(step, taskCtx), nil
	}
}

// NewLLMAgent builds a tool-less llmagent whose system prompt is re-composed
// per-turn via the supplied InstructionProvider. Used for the generation
// sub-agent and for the critic in the M14.2 runtime-instruction pattern
// (per docs/m13/ack-oe-ai-assist-plan-2026-05-17.md).
//
// The model arrives as the adkmodel.LLM interface: production passes the
// gateway-fronted client built in run.go, a test passes a fake.
func NewLLMAgent(name string, m adkmodel.LLM, provider llmagent.InstructionProvider) (agent.Agent, error) {
	a, err := llmagent.New(llmagent.Config{
		Name:                name,
		Model:               m,
		InstructionProvider: provider,
	})
	if err != nil {
		return nil, fmt.Errorf("llmagent.New(%s): %w", name, err)
	}
	return a, nil
}

// questionInstructionProvider wraps the per-turn generation provider in
// ADR-197 M-A prompt stamping, so each composed prompt stamps
// prompt_version + content_hash + the qgen_question conditions onto the
// active span (explainability). WithStamping returns the prompt
// UNCHANGED, so this is behaviour-neutral: the prompt the model sees is
// byte-identical.
func questionInstructionProvider(cfg Config, step qgenagent.Step) llmagent.InstructionProvider {
	return promptstamping.WithStamping(
		cfg.PromptVersion,
		func(s session.ReadonlyState) map[string]string { return qgenagent.QuestionConditions(s) },
		StepInstructionProvider(step),
	)
}

// criticStampedProvider is the critic's ADR-197 M-A counterpart: it stamps
// prompt_version + content_hash + the critic conditions (question_type /
// attempt_index / has_prior_notes) onto the active span. Prompt unchanged.
func criticStampedProvider(cfg Config, step qgenagent.Step) llmagent.InstructionProvider {
	return promptstamping.WithStamping(
		cfg.PromptVersion,
		func(s session.ReadonlyState) map[string]string { return qgenagent.CriticConditions(s) },
		CriticInstructionProvider(step),
	)
}

// allStepsQuestion is the canonical step set. It is a package var rather
// than a direct call so a test can drive the fail-loud guard in
// NewQuestionPipeline. Production always holds the real constructor.
var allStepsQuestion = qgenagent.AllStepsQuestion

// NewQuestionPipeline builds the qgen_question crew: a single-sub-agent
// sequential wrapper around the generation llmagent.
//
// The wrapper is kept even though the crew was trimmed to one sub-agent
// (2026-06-01) so the crew identity "qgen_question" + the
// "qgen_question_generation" event author (which the orchestrator's executor
// terminal-reads) stay stable. Assurance + evaluation dropped: Model Armor
// guardrail_pre covers pre-gen safety; qgen_critic does the quality judging.
func NewQuestionPipeline(cfg Config, llm adkmodel.LLM) (agent.Agent, error) {
	steps := allStepsQuestion()
	if len(steps) != 1 {
		return nil, fmt.Errorf(
			"AllStepsQuestion() returned %d steps; expected 1 (generation-only after the 2026-06-01 trim)",
			len(steps))
	}

	generation, err := NewLLMAgent(steps[0].Name, llm, questionInstructionProvider(cfg, steps[0]))
	if err != nil {
		return nil, err
	}

	pipeline, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name: questionPipelineName,
			Description: "qgen_question single-Q AI-assist generation crew (trimmed " +
				"2026-06-01 to 1 sub-agent): qgen_question_generation (4 prompt templates " +
				"per Intent × QuestionType). Assurance + evaluation dropped. Separate from " +
				"the 6-agent qgen_pipeline batch crew. Serves chora-creation single-Q " +
				"AI-assist (ai_draft + ai_model_answer paths).",
			SubAgents: []agent.Agent{
				generation,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("sequentialagent.New: %w", err)
	}
	return pipeline, nil
}

// NewCriticAgent builds the qgen_critic agent: a single tool-free llmagent
// (P1_SINGLE_AGENT, no internal sub-pipeline). It reviews the candidate JSON
// injected into the instruction from session.State() and emits the critique
// JSON. No external calls.
func NewCriticAgent(cfg Config, llm adkmodel.LLM) (agent.Agent, error) {
	step := qgenagent.CriticStep()
	return NewLLMAgent(step.Name, llm, criticStampedProvider(cfg, step))
}

// Agent tree names of the qgen_question root: the router owns the binary's
// name (the termination + executor-facing identity), the generation pipeline
// keeps its sub-agent qgen_question_generation as the event author the
// kennel's executor terminal-reads, the composer is a two-step sequence.
const (
	questionPipelineName = "qgen_question_pipeline"
	composeAgentName     = "qgen_question_compose"
	composeFinaliseName  = "qgen_question_compose_finalise"
)

// composeBeforeModel replaces the trigger user content with the composer's
// user parts: the instruction text, the sources as object-store FileData and
// the rubric behind its marker (the kennel's _build_contents_json by reference).
func composeBeforeModel(ctx agent.CallbackContext, req *adkmodel.LLMRequest) (*adkmodel.LLMResponse, error) {
	creq, err := ParseComposeRequest(ctx.State())
	if err != nil {
		return nil, err
	}
	parts := ComposeUserParts(creq)
	if req == nil {
		return nil, nil
	}
	for i := len(req.Contents) - 1; i >= 0; i-- {
		if req.Contents[i] != nil && req.Contents[i].Role == "user" {
			req.Contents[i].Parts = parts
			return nil, nil
		}
	}
	req.Contents = append(req.Contents, &genai.Content{Role: "user", Parts: parts})
	return nil, nil
}

// composeConditions surfaces the prompt discriminants (ADR-197 M-A).
func composeConditions(st interface {
	Get(string) (any, error)
}) map[string]string {
	cond := map[string]string{"mode": ModeCompose}
	if creq, err := ParseComposeRequest(st); err == nil {
		sources, rubric := SplitSourceFiles(creq.SourceFiles)
		cond["grounded"] = fmt.Sprint(len(sources) > 0)
		cond["rubric"] = fmt.Sprint(rubric != nil)
	}
	return cond
}

// finaliseCompose parses the composer's raw answer and renders the envelope.
func finaliseCompose(st interface {
	Get(string) (any, error)
}) (string, error) {
	creq, err := ParseComposeRequest(st)
	if err != nil {
		return "", err
	}
	parsed, err := ParseComposeAnswer(StateString(st, stateKeyComposeRaw))
	if err != nil {
		return "", agentdispatch.Permanent(err.Error(), err)
	}
	return ComposeEnvelope(NormaliseProposal(parsed, creq)), nil
}

// NewComposeAgent builds the mode=compose sequence: the composer llmagent
// (system prompt + the user parts, OutputKey) then the deterministic
// finaliser whose text event is the completion.
func NewComposeAgent(cfg Config, composeLLM adkmodel.LLM) (agent.Agent, error) {
	if composeLLM == nil {
		return nil, fmt.Errorf("NewComposeAgent: the compose LLM is required")
	}
	composer, err := llmagent.New(llmagent.Config{
		Name:      composeAgentName,
		Model:     composeLLM,
		OutputKey: stateKeyComposeRaw,
		InstructionProvider: promptstamping.WithStamping(cfg.ComposePromptVersion,
			func(s session.ReadonlyState) map[string]string { return composeConditions(s) },
			func(ctx agent.ReadonlyContext) (string, error) {
				if _, err := ParseComposeRequest(ctx.ReadonlyState()); err != nil {
					return "", err
				}
				return ComposeSystemPrompt, nil
			}),
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{composeBeforeModel},
	})
	if err != nil {
		return nil, fmt.Errorf("llmagent.New(%s): %w", composeAgentName, err)
	}
	finaliser, err := agent.New(agent.Config{
		Name:        composeFinaliseName,
		Description: "normalises the composer's proposal into the ProposedTestSet contract",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				text, err := finaliseCompose(ctx.Session().State())
				if err != nil {
					yield(nil, err)
					return
				}
				ev := session.NewEvent(ctx.InvocationID())
				ev.Author = composeFinaliseName
				ev.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		return nil, fmt.Errorf("agent.New(%s): %w", composeFinaliseName, err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
		Name:        composeAgentName + "_seq",
		Description: "qgen_generate mode=compose: composer then finaliser",
		SubAgents:   []agent.Agent{composer, finaliser},
	}})
	if err != nil {
		return nil, fmt.Errorf("sequentialagent.New(compose): %w", err)
	}
	return seq, nil
}

// NewQuestionRoot builds the qgen_question root: a mode router over the
// generation pipeline (mode absent | generate) and the composer (compose);
// an unknown mode is a permanent failure raised before any model call. A nil
// composeLLM leaves compose refused by name (the binary then serves generate
// only), never silently routed to generation.
func NewQuestionRoot(cfg Config, llm, composeLLM adkmodel.LLM) (agent.Agent, error) {
	pipeline, err := NewQuestionPipeline(cfg, llm)
	if err != nil {
		return nil, err
	}
	subAgents := []agent.Agent{pipeline}
	var compose agent.Agent
	if composeLLM != nil {
		compose, err = NewComposeAgent(cfg, composeLLM)
		if err != nil {
			return nil, err
		}
		subAgents = append(subAgents, compose)
	}
	return agent.New(agent.Config{
		Name:        CrewKindQuestion,
		Description: "qgen_generate mode router: generate (question pipeline) | compose (testset composer)",
		SubAgents:   subAgents,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				mode, err := selectMode(ctx.Session().State())
				if err != nil {
					yield(nil, err)
					return
				}
				child := pipeline
				if mode == ModeCompose {
					if compose == nil {
						yield(nil, agentdispatch.Permanent("unknown_mode: compose is not served by this build (no composer model configured)", nil))
						return
					}
					child = compose
				}
				for ev, err := range child.Run(ctx) {
					if !yield(ev, err) {
						return
					}
					if err != nil {
						return
					}
				}
			}
		},
	})
}
