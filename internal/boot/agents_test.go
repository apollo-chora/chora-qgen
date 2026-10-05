// Tests for the M14.2 per-turn instruction providers and the agent
// construction they feed.
//
// The provider tests moved here verbatim from cmd/qgen_question/main_test.go
// when the boot wiring was extracted out of main(); only the call sites
// changed from the package-private stepInstructionProvider to the exported
// StepInstructionProvider. The provider is invoked by the ADK runtime on
// EVERY model turn: it reads session.ReadonlyState (set by the Python
// executor's _build_session_state) and folds intent / question_type /
// author_* into the system prompt via the existing pure-function composer.
//
// The fakes live in fakes_test.go.
package boot

import (
	"errors"
	"github.com/apollo-chora/chora-adk-common/agentdispatch"
	"strings"
	"testing"

	qgenagent "github.com/apollo-chora/chora-qgen/internal/agent"
)

// ---------------------------------------------------------------------------
// Per-turn instruction provider: the load-bearing M14.2 contract.
// ---------------------------------------------------------------------------

func TestStepInstructionProvider_isCallable_andReturnsCREATEBlock(t *testing.T) {
	// Sanity: the provider returns a non-empty CREATE-block prompt for
	// the canonical generation step.
	provider := StepInstructionProvider(qgenagent.StepGeneration3())
	if provider == nil {
		t.Fatalf("StepInstructionProvider returned nil: agents would have no system prompt")
	}
	got, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "mcq",
	}))
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if got == "" {
		t.Errorf("provider returned empty instruction")
	}
	// The CREATE blocks MUST be there, the same contract the pure-composer
	// tests assert.
	for _, want := range []string{"[CONTEXT]", "[ROLE]", "[TASK]", "[EXPECTED OUTPUT]"} {
		if !strings.Contains(got, want) {
			t.Errorf("provider output missing CREATE block %q", want)
		}
	}
}

func TestStepInstructionProvider_OEStateProducesNestedOEPayload(t *testing.T) {
	// State carrying question_type=oe MUST drive the runtime composer
	// into the new_oe template (nested oe_payload + rubric + grader_tier).
	// This is the assertion the OE smoke 4/4 hinges on. Prior to M14.2
	// the boot-time hardcode emitted the new_mcq template no matter what
	// state said.
	provider := StepInstructionProvider(qgenagent.StepGeneration3())
	got, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "oe",
		"input_payload": "Explain photosynthesis",
	}))
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	for _, want := range []string{"oe_payload", "rubric", "grader_tier", "40 words"} {
		if !strings.Contains(got, want) {
			t.Errorf("OE-from-state instruction missing %q\n--- prompt:\n%s", want, got)
		}
	}
}

func TestStepInstructionProvider_MCQStateProducesMCQTemplate(t *testing.T) {
	// State carrying question_type=mcq drives the new_mcq template
	// (options + distractors + per-option explainers). Companion to the
	// OE assertion so we cover both halves of the runtime branching.
	provider := StepInstructionProvider(qgenagent.StepGeneration3())
	got, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "mcq",
		"input_payload": "Generate an MCQ on graph algorithms",
	}))
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if !strings.Contains(got, "options") {
		t.Errorf("MCQ instruction missing 'options'")
	}
	if !strings.Contains(strings.ToLower(got), "distractor") {
		t.Errorf("MCQ instruction missing 'distractor'")
	}
	if strings.Contains(got, "oe_payload") || strings.Contains(got, "grader_tier") {
		t.Errorf("MCQ instruction contains OE phrasing, branching broken")
	}
}

func TestStepInstructionProvider_modelAnswerFillPreservesAuthorStem(t *testing.T) {
	// intent=model_answer_fill + question_type=mcq + author_* surfaces
	// the author's verbatim stem in the [TASK] block. Mirrors
	// TestComposeQGenQuestionInstruction_runtimeModelAnswerFillPreservesAuthorStem
	// from composer_question_test.go but exercises it via the runtime
	// provider closure path the live engine uses.
	authorStem := "What is the runtime complexity of binary search?"
	provider := StepInstructionProvider(qgenagent.StepGeneration3())
	got, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "model_answer_fill",
		"question_type": "mcq",
		"author_stem":   authorStem,
		"author_options": []any{
			map[string]any{"option_id": "opt-1", "label": "O(n)", "is_correct": false},
			map[string]any{"option_id": "opt-2", "label": "O(log n)", "is_correct": true},
		},
	}))
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if !strings.Contains(got, authorStem) {
		t.Errorf("fill-MCQ instruction does not surface the author's verbatim stem %q", authorStem)
	}
	if !strings.Contains(got, "O(log n)") {
		t.Errorf("fill-MCQ instruction does not surface the author's correct-option label")
	}
}

func TestStepInstructionProvider_callbackInvokedPerTurn(t *testing.T) {
	// The ADK runtime invokes the InstructionProvider on EVERY turn, so we
	// simulate two consecutive calls with DIFFERENT state and assert the
	// returned instructions differ. This is the contract that distinguishes
	// the M14.2 fix from the prior boot-time compose (which would emit
	// the SAME prompt regardless of state).
	provider := StepInstructionProvider(qgenagent.StepGeneration3())

	mcqInstruction, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "mcq",
	}))
	if err != nil {
		t.Fatalf("mcq provider call: %v", err)
	}

	oeInstruction, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "oe",
	}))
	if err != nil {
		t.Fatalf("oe provider call: %v", err)
	}

	if mcqInstruction == oeInstruction {
		t.Errorf("provider returned identical instructions for MCQ vs OE state, per-turn recomposition is broken")
	}
}

func TestStepInstructionProvider_invalidQuestionTypeFailsLoud(t *testing.T) {
	// Anything outside {mcq, oe} MUST fail the turn at the provider: the
	// fail-loud posture is what differentiates M14.2 from the prior
	// silent-MCQ-coercion bug.
	provider := StepInstructionProvider(qgenagent.StepGeneration3())
	_, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"question_type": "essay",
	}))
	if err == nil {
		t.Errorf("provider accepted invalid question_type \"essay\", should fail loud")
	}
}

func TestStepInstructionProvider_assuranceCarriesIntentCue(t *testing.T) {
	// The assurance step's [ROLE] block depends on the intent cue (so it
	// can apply the right downstream expectations). With per-turn
	// composition the cue now reflects the actual runtime intent rather
	// than the boot-time hardcoded "new_question".
	provider := StepInstructionProvider(qgenagent.StepAssurance3())
	got, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "model_answer_fill",
		"question_type": "oe",
	}))
	if err != nil {
		t.Fatalf("provider call: %v", err)
	}
	if !strings.Contains(got, "model_answer_fill") {
		t.Errorf("assurance instruction missing runtime intent cue \"model_answer_fill\"")
	}
}

// ---------------------------------------------------------------------------
// CriticInstructionProvider
// ---------------------------------------------------------------------------

func TestCriticInstructionProvider_injectsTheCandidateFromState(t *testing.T) {
	// The candidate is the whole point: before the M14.2 fix the critic
	// composed a boot-time prompt with placeholder text, never saw the
	// candidate, and answered "input_unparseable" on every job.
	candidate := `{"stem":"What is 2+2?","options":[{"label":"4","is_correct":true}]}`
	provider := CriticInstructionProvider(qgenagent.CriticStep())
	if provider == nil {
		t.Fatalf("CriticInstructionProvider returned nil")
	}
	got, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"question_type": "mcq",
		"input_payload": candidate,
	}))
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if !strings.Contains(got, candidate) {
		t.Errorf("critic instruction does not carry the candidate JSON from state\n--- prompt:\n%s", got)
	}
	for _, want := range []string{"[CONTEXT]", "[ROLE]", "[TASK]", "[EXPECTED OUTPUT]"} {
		if !strings.Contains(got, want) {
			t.Errorf("critic instruction missing CREATE block %q", want)
		}
	}
}

func TestCriticInstructionProvider_missingCandidateFailsLoud(t *testing.T) {
	// An empty input_payload is a pipeline bug upstream. The provider must
	// surface it rather than silently critiquing nothing.
	provider := CriticInstructionProvider(qgenagent.CriticStep())
	_, err := provider(newFakeCtx(map[string]any{
		"tenant_id":     "tenant-x",
		"question_type": "mcq",
	}))
	if err == nil {
		t.Fatal("critic provider accepted a state with no candidate, should fail loud")
	}
	if !strings.Contains(err.Error(), "input_payload") {
		t.Errorf("critic error does not name the missing state key: %v", err)
	}
}

func TestCriticInstructionProvider_recomposesPerTurn(t *testing.T) {
	provider := CriticInstructionProvider(qgenagent.CriticStep())

	mcq, err := provider(newFakeCtx(map[string]any{
		"question_type": "mcq",
		"input_payload": `{"stem":"q"}`,
	}))
	if err != nil {
		t.Fatalf("mcq provider call: %v", err)
	}
	oe, err := provider(newFakeCtx(map[string]any{
		"question_type": "oe",
		"input_payload": `{"stem":"q"}`,
	}))
	if err != nil {
		t.Fatalf("oe provider call: %v", err)
	}
	if mcq == oe {
		t.Errorf("critic provider returned identical instructions for MCQ vs OE state, per-turn recomposition is broken")
	}
}

// ---------------------------------------------------------------------------
// Agent construction: the adkmodel.LLM injection seam
// ---------------------------------------------------------------------------

func TestNewLLMAgent_carriesTheStepName(t *testing.T) {
	a, err := NewLLMAgent("qgen_question_generation", &fakeLLM{name: "fake"},
		StepInstructionProvider(qgenagent.StepGeneration3()))
	if err != nil {
		t.Fatalf("NewLLMAgent: %v", err)
	}
	if a.Name() != "qgen_question_generation" {
		t.Errorf("agent name = %q, want %q", a.Name(), "qgen_question_generation")
	}
	if len(a.SubAgents()) != 0 {
		t.Errorf("llmagent has %d sub-agents, want 0 (it is a leaf)", len(a.SubAgents()))
	}
}

func TestNewQuestionPipeline_shapeAndIdentity(t *testing.T) {
	cfg := Config{Binary: CrewKindQuestion, PromptVersion: "v1"}
	pipeline, err := NewQuestionPipeline(cfg, &fakeLLM{name: "fake"})
	if err != nil {
		t.Fatalf("NewQuestionPipeline: %v", err)
	}

	// The crew identity is terminal-read by the Python orchestrator's
	// executor. Renaming it breaks the orchestrator, not just this binary.
	if pipeline.Name() != questionPipelineName {
		t.Errorf("pipeline name = %q, want %q", pipeline.Name(), questionPipelineName)
	}

	subs := pipeline.SubAgents()
	if len(subs) != 1 {
		t.Fatalf("pipeline has %d sub-agents, want 1 (generation-only after the 2026-06-01 trim)", len(subs))
	}
	// The event author the orchestrator terminal-reads.
	if subs[0].Name() != "qgen_question_generation" {
		t.Errorf("sub-agent name = %q, want %q", subs[0].Name(), "qgen_question_generation")
	}
	if subs[0].Name() != qgenagent.AllStepsQuestion()[0].Name {
		t.Errorf("sub-agent name %q drifted from AllStepsQuestion()[0].Name %q",
			subs[0].Name(), qgenagent.AllStepsQuestion()[0].Name)
	}

	// The crew description is the human-readable contract other crews and
	// the registry read; pin the load-bearing clauses.
	for _, want := range []string{
		"qgen_question single-Q AI-assist generation crew",
		"qgen_question_generation",
		"Assurance + evaluation dropped",
		"6-agent qgen_pipeline batch crew",
	} {
		if !strings.Contains(pipeline.Description(), want) {
			t.Errorf("pipeline description missing %q\n--- got:\n%s", want, pipeline.Description())
		}
	}

	if pipeline.FindSubAgent("qgen_question_generation") == nil {
		t.Errorf("pipeline cannot find its own generation sub-agent by name")
	}
}

func TestNewQuestionPipeline_failsLoudWhenTheStepSetChanges(t *testing.T) {
	// The crew was trimmed to one sub-agent on 2026-06-01. Re-adding a step
	// without updating the pipeline would silently drop it from the
	// sequential wrapper, so the guard must refuse to build instead.
	original := allStepsQuestion
	t.Cleanup(func() { allStepsQuestion = original })

	allStepsQuestion = func() []qgenagent.Step {
		return []qgenagent.Step{qgenagent.StepAssurance3(), qgenagent.StepGeneration3()}
	}

	_, err := NewQuestionPipeline(Config{Binary: CrewKindQuestion, PromptVersion: "v1"}, &fakeLLM{name: "fake"})
	if err == nil {
		t.Fatal("NewQuestionPipeline built a pipeline from 2 steps; the fail-loud guard is gone")
	}
	if !strings.Contains(err.Error(), "expected 1") {
		t.Errorf("guard error does not explain the expectation: %v", err)
	}

	// And the empty case, which would otherwise index out of range.
	allStepsQuestion = func() []qgenagent.Step { return nil }
	if _, err := NewQuestionPipeline(Config{Binary: CrewKindQuestion}, &fakeLLM{name: "fake"}); err == nil {
		t.Fatal("NewQuestionPipeline accepted an empty step set")
	}
}

func TestNewQuestionPipeline_subAgentPromptStillRecomposesPerTurn(t *testing.T) {
	// promptstamping.WithStamping must wrap the provider WITHOUT altering
	// the prompt. Construct the pipeline, then drive the same underlying
	// provider the sub-agent holds and assert it still branches on state.
	cfg := Config{Binary: CrewKindQuestion, PromptVersion: "v1"}
	if _, err := NewQuestionPipeline(cfg, &fakeLLM{name: "fake"}); err != nil {
		t.Fatalf("NewQuestionPipeline: %v", err)
	}

	stamped := questionInstructionProvider(cfg, qgenagent.AllStepsQuestion()[0])
	plain := StepInstructionProvider(qgenagent.AllStepsQuestion()[0])

	state := map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "oe",
	}
	gotStamped, err := stamped(newFakeCtx(state))
	if err != nil {
		t.Fatalf("stamped provider: %v", err)
	}
	gotPlain, err := plain(newFakeCtx(state))
	if err != nil {
		t.Fatalf("plain provider: %v", err)
	}
	if gotStamped != gotPlain {
		t.Errorf("promptstamping altered the prompt; it must be behaviour-neutral")
	}
}

func TestNewCriticAgent_identity(t *testing.T) {
	cfg := Config{Binary: CrewKindCritic, PromptVersion: "v1"}
	critic, err := NewCriticAgent(cfg, &fakeLLM{name: "fake"})
	if err != nil {
		t.Fatalf("NewCriticAgent: %v", err)
	}
	if critic.Name() != "qgen_critic" {
		t.Errorf("critic name = %q, want %q", critic.Name(), "qgen_critic")
	}
	if critic.Name() != qgenagent.CriticStep().Name {
		t.Errorf("critic name %q drifted from CriticStep().Name %q", critic.Name(), qgenagent.CriticStep().Name)
	}
	// P1_SINGLE_AGENT: no internal sub-pipeline.
	if len(critic.SubAgents()) != 0 {
		t.Errorf("critic has %d sub-agents, want 0 (P1 single agent)", len(critic.SubAgents()))
	}
}

func TestNewCriticAgent_promptStampingIsBehaviourNeutral(t *testing.T) {
	cfg := Config{Binary: CrewKindCritic, PromptVersion: "v1"}
	stamped := criticStampedProvider(cfg, qgenagent.CriticStep())
	plain := CriticInstructionProvider(qgenagent.CriticStep())

	state := map[string]any{
		"tenant_id":     "tenant-x",
		"question_type": "oe",
		"input_payload": `{"stem":"Explain photosynthesis"}`,
	}
	gotStamped, err := stamped(newFakeCtx(state))
	if err != nil {
		t.Fatalf("stamped provider: %v", err)
	}
	gotPlain, err := plain(newFakeCtx(state))
	if err != nil {
		t.Fatalf("plain provider: %v", err)
	}
	if gotStamped != gotPlain {
		t.Errorf("promptstamping altered the critic prompt; it must be behaviour-neutral")
	}
}

func TestSelectMode(t *testing.T) {
	var perm *agentdispatch.PermanentError
	for mode, want := range map[string]string{"": ModeGenerate, "generate": ModeGenerate, " Generate ": ModeGenerate, "compose": ModeCompose} {
		got, err := selectMode(mapState{"mode": mode})
		if err != nil || got != want {
			t.Fatalf("mode %q: got %q err %v", mode, got, err)
		}
	}
	if _, err := selectMode(mapState{"mode": "render"}); !errors.As(err, &perm) || !strings.Contains(err.Error(), "unknown_mode: render") {
		t.Fatalf("unknown mode must be permanent: %v", err)
	}
}

type mapState map[string]any

func (m mapState) Get(k string) (any, error) {
	if v, ok := m[k]; ok {
		return v, nil
	}
	return nil, errors.New("missing")
}
