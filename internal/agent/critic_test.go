// Tests for the qgen_critic prompt composer.
//
// RED→GREEN→REFACTOR per [[feedback-strict-tdd]] — these tests run against
// the pure-function composer in critic.go. No LLM, no model call, no event
// bus — fast unit tests covering:
//
//   - 6-block CREATE prompt structure (CONTEXT / ROLE / EXAMPLES / AUDIENCE /
//     TASK / EXPECTED OUTPUT)
//   - Per-QuestionType TaskBlock + OutputBlock branching (mcq vs oe)
//   - PriorCriticNotes surfacing in [CONTEXT] on retries (AttemptIndex > 0)
//   - Metadata hints (subject / cognitive_level / difficulty) surfacing
//   - Unsupported question_type short-circuit (reserved sentinels)
//   - Anti-pattern guards (no numeric scoring; no fabrication; JSON-only)
//   - Deterministic / pure (same input → same output)
package agent

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

func newMCQCriticCtx() CriticTaskContext {
	return CriticTaskContext{
		TenantID:           "tenant-aaa",
		AuthorGCID:         "gcid-bbb",
		JobID:              "job-ccc",
		QuestionType:       QuestionTypeMCQ,
		AttemptIndex:       0,
		MaxAttempts:        4,
		AuthorPrompt:       "Generate a question on photosynthesis byproducts",
		SubjectHint:        "Biology",
		CognitiveLevelHint: "knowledge",
		DifficultyHint:     "1",
		CandidateJSON:      `{"stem":"During the light-dependent reactions of photosynthesis, which gas is released as a byproduct?","question_type":"mcq","mcq_payload":{"options":[{"option_id":"a","label":"A","text":"Oxygen","is_correct":true,"explainer":"Water is split, releasing O2."},{"option_id":"b","label":"B","text":"Carbon dioxide","is_correct":false,"explainer":"CO2 is consumed, not released."}],"scoring_mode":"single_correct"}}`,
	}
}

func newOECriticCtx() CriticTaskContext {
	c := newMCQCriticCtx()
	c.QuestionType = QuestionTypeOE
	c.AuthorPrompt = "Explain chlorophyll in 2-3 sentences"
	c.CognitiveLevelHint = "comprehension"
	c.DifficultyHint = "3"
	return c
}

// -----------------------------------------------------------------------------
// Candidate embedding — the M14.2 critic-instruction fix (2026-06-01)
//
// The prior boot-time static compose never put the per-request candidate in
// front of the critic (the InstructionProvider was deferred + never built),
// so the critic always answered "input_unparseable: no candidate provided"
// and the orchestrator quality loop maxed out every job. These tests pin the
// fix: ComposeCriticInstruction MUST embed CandidateJSON, and
// BuildCriticTaskContextFromState MUST read it (+ the per-turn context) from
// session.State() and fail loud when the candidate is missing.
// -----------------------------------------------------------------------------

func TestComposeCriticInstruction_EmbedsCandidate(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	if !strings.Contains(got, "[CANDIDATE TO CRITIQUE]") {
		t.Errorf("composed instruction missing the [CANDIDATE TO CRITIQUE] block;\n%s", got)
	}
	// The verbatim candidate JSON must appear so the LLM actually critiques it
	// (the user message is only a 'BEGIN' trigger — the candidate rides in the
	// instruction via session.state, NOT the message).
	if !strings.Contains(got, "light-dependent reactions of photosynthesis") {
		t.Errorf("composed instruction did not embed the candidate JSON;\n%s", got)
	}
}

func TestBuildCriticTaskContextFromState_ReadsCandidateAndContext(t *testing.T) {
	t.Parallel()
	candidate := `{"stem":"x","question_type":"mcq","mcq_payload":{"options":[]}}`
	st := &fakeState{data: map[string]any{
		"tenant_id":          "tenant-xyz",
		"author_gcid":        "gcid-123",
		"job_id":             "job-777",
		"question_type":      "mcq",
		"attempt_index":      float64(2), // JSON numbers arrive as float64
		"max_attempts":       float64(4),
		"prior_critic_notes": "H1: only 2 options",
		"author_prompt":      "make an MCQ about cell division",
		"input_payload":      candidate,
	}}
	ctx, err := BuildCriticTaskContextFromState(st)
	if err != nil {
		t.Fatalf("BuildCriticTaskContextFromState: unexpected error: %v", err)
	}
	if ctx.CandidateJSON != candidate {
		t.Errorf("CandidateJSON = %q, want the input_payload candidate", ctx.CandidateJSON)
	}
	if ctx.TenantID != "tenant-xyz" || ctx.AuthorGCID != "gcid-123" || ctx.JobID != "job-777" {
		t.Errorf("identity fields not read: %+v", ctx)
	}
	if ctx.QuestionType != QuestionTypeMCQ {
		t.Errorf("QuestionType = %q, want mcq", ctx.QuestionType)
	}
	if ctx.AttemptIndex != 2 || ctx.MaxAttempts != 4 {
		t.Errorf("attempt/max not read: index=%d max=%d", ctx.AttemptIndex, ctx.MaxAttempts)
	}
	if ctx.PriorCriticNotes != "H1: only 2 options" {
		t.Errorf("PriorCriticNotes = %q", ctx.PriorCriticNotes)
	}
	if ctx.AuthorPrompt != "make an MCQ about cell division" {
		t.Errorf("AuthorPrompt = %q", ctx.AuthorPrompt)
	}
}

func TestBuildCriticTaskContextFromState_AuthorGCIDFallsBackToUserGCID(t *testing.T) {
	t.Parallel()
	st := &fakeState{data: map[string]any{
		"tenant_id":     "t",
		"user_gcid":     "gcid-fallback",
		"question_type": "mcq",
		"input_payload": `{"stem":"x"}`,
	}}
	ctx, err := BuildCriticTaskContextFromState(st)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ctx.AuthorGCID != "gcid-fallback" {
		t.Errorf("AuthorGCID fallback = %q, want gcid-fallback", ctx.AuthorGCID)
	}
}

func TestBuildCriticTaskContextFromState_MissingCandidateFailsLoud(t *testing.T) {
	t.Parallel()
	st := &fakeState{data: map[string]any{
		"tenant_id":     "t",
		"author_gcid":   "g",
		"question_type": "mcq",
		// input_payload intentionally absent
	}}
	if _, err := BuildCriticTaskContextFromState(st); err == nil {
		t.Fatal("expected a fail-loud error when the candidate is missing, got nil")
	}
}

func TestBuildCriticTaskContextFromState_InvalidQuestionTypeFailsLoud(t *testing.T) {
	t.Parallel()
	st := &fakeState{data: map[string]any{
		"tenant_id":     "t",
		"author_gcid":   "g",
		"question_type": "essay", // not mcq/oe
		"input_payload": `{"stem":"x"}`,
	}}
	if _, err := BuildCriticTaskContextFromState(st); err == nil {
		t.Fatal("expected a fail-loud error for an invalid question_type, got nil")
	}
}

func TestBuildCriticTaskContextFromState_NilStateFailsLoud(t *testing.T) {
	t.Parallel()
	if _, err := BuildCriticTaskContextFromState(nil); err == nil {
		t.Fatal("expected an error for nil state, got nil")
	}
}

// -----------------------------------------------------------------------------
// Step descriptor — invariants
// -----------------------------------------------------------------------------

func TestCriticStep_Name(t *testing.T) {
	t.Parallel()
	step := CriticStep()
	if step.Name != "qgen_critic" {
		t.Fatalf("step.Name = %q; want qgen_critic", step.Name)
	}
}

func TestCriticStep_PlaceholderTaskBlock(t *testing.T) {
	t.Parallel()
	step := CriticStep()
	// Per the 3-agent composer convention the static Step descriptor leaves
	// TaskBlock + OutputBlock as placeholders; ComposeCriticInstruction
	// resolves them per QuestionType.
	if !strings.Contains(step.TaskBlock, "resolved per QuestionType") {
		t.Errorf("step.TaskBlock should be a placeholder; got %q", step.TaskBlock)
	}
	if !strings.Contains(step.OutputBlock, "resolved per QuestionType") {
		t.Errorf("step.OutputBlock should be a placeholder; got %q", step.OutputBlock)
	}
}

func TestCriticStep_RoleBlockDistinguishesFromSixAgentEvaluator(t *testing.T) {
	t.Parallel()
	step := CriticStep()
	// Per user direction 2026-05-17 — the name+role must distinguish from
	// the 6-agent crew's `evaluator` (scoring) to avoid label clash.
	if !strings.Contains(step.RoleBlock, "Critic") {
		t.Errorf("RoleBlock should self-identify as Critic; got %q", step.RoleBlock)
	}
	if !strings.Contains(step.RoleBlock, "NOT scoring") {
		t.Errorf("RoleBlock should explicitly disclaim scoring; got %q", step.RoleBlock)
	}
	if !strings.Contains(step.RoleBlock, "6-agent") {
		t.Errorf("RoleBlock should distinguish from the 6-agent evaluator; got %q", step.RoleBlock)
	}
}

// -----------------------------------------------------------------------------
// Composer — 6-block CREATE prompt structure
// -----------------------------------------------------------------------------

func TestComposeCriticInstruction_HasAllSixBlocks(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	wantBlocks := []string{
		"## [CONTEXT]",
		"## [ROLE]",
		"## [EXAMPLES]",
		"## [AUDIENCE]",
		"## [TASK]",
		"## [EXPECTED OUTPUT]",
	}
	for _, b := range wantBlocks {
		if !strings.Contains(got, b) {
			t.Errorf("composed prompt missing block %q", b)
		}
	}
}

func TestComposeCriticInstruction_ContextHasCallAttributes(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	// Span attribution + IMDA D1 accountability needs tenant/author/job in scope.
	wantStrings := []string{
		"tenant_id=tenant-aaa",
		"author_gcid=gcid-bbb",
		"job_id=job-ccc",
		"question_type=mcq",
		"attempt=1/4",
		"Generate a question on photosynthesis byproducts",
	}
	for _, s := range wantStrings {
		if !strings.Contains(got, s) {
			t.Errorf("composed prompt missing %q\n--- prompt ---\n%s", s, got)
		}
	}
}

func TestComposeCriticInstruction_MetadataHintsSurface(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	// All three hint fields should reach [CONTEXT].
	for _, want := range []string{"subject=Biology", "cognitive_level=knowledge", "difficulty=1"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing metadata hint %q", want)
		}
	}
}

func TestComposeCriticInstruction_MetadataHintsOmittedWhenEmpty(t *testing.T) {
	t.Parallel()
	ctx := newMCQCriticCtx()
	ctx.SubjectHint = ""
	ctx.CognitiveLevelHint = ""
	ctx.DifficultyHint = ""
	got := ComposeCriticInstruction(CriticStep(), ctx)
	if strings.Contains(got, "Metadata hints:") {
		t.Errorf("Metadata hints line should be absent when all hints empty; got prompt with line\n%s", got)
	}
}

// -----------------------------------------------------------------------------
// Quality-loop affordances — PriorCriticNotes on retries
// -----------------------------------------------------------------------------

func TestComposeCriticInstruction_PriorCriticNotesSurfaceOnRetry(t *testing.T) {
	t.Parallel()
	ctx := newMCQCriticCtx()
	ctx.AttemptIndex = 2 // 3rd attempt (0-based)
	ctx.PriorCriticNotes = "option_id=b distractor is a true statement; option_id=c too implausible"
	got := ComposeCriticInstruction(CriticStep(), ctx)
	if !strings.Contains(got, "attempt=3/4") {
		t.Errorf("attempt counter mis-rendered; want attempt=3/4 in %s", got)
	}
	if !strings.Contains(got, "Prior critic_notes") {
		t.Errorf("Prior critic_notes line should appear on retries; got %s", got)
	}
	if !strings.Contains(got, "option_id=b distractor is a true statement") {
		t.Errorf("PriorCriticNotes content should be surfaced verbatim")
	}
}

func TestComposeCriticInstruction_UnsetMaxAttemptsRendersDefaultBudget(t *testing.T) {
	t.Parallel()
	// An orchestrator that did not stamp max_attempts (state key absent reads
	// as 0) must still show the critic a real budget. 0 is the documented
	// default of 4: rendering "attempt=1/0" would tell the critic it is past
	// the last retry on the very first pass and skew its leniency.
	ctx := newMCQCriticCtx()
	ctx.MaxAttempts = 0
	got := ComposeCriticInstruction(CriticStep(), ctx)
	if !strings.Contains(got, "attempt=1/4") {
		t.Errorf("unset max_attempts must render the default budget of 4; got %s", got)
	}
	if strings.Contains(got, "attempt=1/0") {
		t.Errorf("a zero budget must never reach the prompt; got %s", got)
	}
}

func TestComposeCriticInstruction_ExplicitMaxAttemptsOverridesDefault(t *testing.T) {
	t.Parallel()
	// Positive control for the guard above: a stamped budget is honoured
	// verbatim, so the default only applies to the unset case.
	ctx := newMCQCriticCtx()
	ctx.MaxAttempts = 2
	got := ComposeCriticInstruction(CriticStep(), ctx)
	if !strings.Contains(got, "attempt=1/2") {
		t.Errorf("an explicit max_attempts must be rendered as-is; got %s", got)
	}
}

func TestComposeCriticInstruction_PriorCriticNotesAbsentOnFirstAttempt(t *testing.T) {
	t.Parallel()
	ctx := newMCQCriticCtx()
	// AttemptIndex=0 + empty PriorCriticNotes — the line MUST NOT appear.
	got := ComposeCriticInstruction(CriticStep(), ctx)
	if strings.Contains(got, "Prior critic_notes") {
		t.Errorf("Prior critic_notes line should be absent on first attempt with empty notes")
	}
}

// -----------------------------------------------------------------------------
// Per-QuestionType branching — mcq vs oe rubric
// -----------------------------------------------------------------------------

func TestComposeCriticInstruction_MCQTaskBlockMentionsMCQConcerns(t *testing.T) {
	t.Parallel()
	// Target the resolved TaskBlock directly (the [EXAMPLES] section is
	// shared MCQ+OE-neutral content and may reference both types).
	task, _ := resolveCriticTaskAndOutput(newMCQCriticCtx())
	for _, want := range []string{
		"Distractor plausibility",
		"single_correct",
		"option_id",
		"mcq_payload",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("MCQ TaskBlock should mention %q; missing", want)
		}
	}
	// Negative: MCQ TaskBlock should not pull in OE-specific rubric-weights
	// concerns (MCQ has no rubric in the OpenAPI schema).
	if strings.Contains(task, "Rubric coverage") {
		t.Errorf("MCQ TaskBlock should not include OE rubric-coverage check")
	}
	if strings.Contains(task, "weights sum to 100") {
		t.Errorf("MCQ TaskBlock should not include OE rubric-weights-sum check")
	}
}

func TestComposeCriticInstruction_OETaskBlockMentionsOEConcerns(t *testing.T) {
	t.Parallel()
	task, _ := resolveCriticTaskAndOutput(newOECriticCtx())
	for _, want := range []string{
		"Model answer accuracy",
		"Rubric coverage",
		// Tightened 2026-05-17 — accepts both 1.0±1% (fractional) and 100±1%
		// (percentage) conventions; the prior "weights sum to 100" hard
		// language is now in H3 of the hard-rejection block.
		"Rubric weights sum tolerance",
		"model_answer",
		"grader_tier",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("OE TaskBlock should mention %q; missing", want)
		}
	}
	// Negative: OE TaskBlock should not pull in MCQ-specific distractor
	// or single/multi-correct integrity concerns.
	if strings.Contains(task, "Distractor plausibility") {
		t.Errorf("OE TaskBlock should not include MCQ-only distractor check")
	}
	if strings.Contains(task, "Single/multi-correct integrity") {
		t.Errorf("OE TaskBlock should not include MCQ-only single/multi-correct integrity check")
	}
}

func TestComposeCriticInstruction_UnsupportedQuestionTypeRefuses(t *testing.T) {
	t.Parallel()
	ctx := newMCQCriticCtx()
	ctx.QuestionType = QuestionType("flashcard") // reserved sentinel — not Phyllis scope
	got := ComposeCriticInstruction(CriticStep(), ctx)
	if !strings.Contains(got, "unsupported_question_type") {
		t.Errorf("unsupported question_type should short-circuit to refuse template; got\n%s", got)
	}
	if !strings.Contains(got, "invalid question_type") {
		t.Errorf("TaskBlock should explicitly refuse on unsupported type")
	}
}

// -----------------------------------------------------------------------------
// Anti-pattern guards (per [TASK] / [EXPECTED OUTPUT])
// -----------------------------------------------------------------------------

func TestComposeCriticInstruction_AntiPatternsListed(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	// User clarification 2026-05-17 — qualitative critic, NOT scoring.
	if !strings.Contains(got, "NEVER score numerically") {
		t.Errorf("anti-pattern: never-score-numerically guard missing")
	}
	// Anti-fabrication — the critic reviews, doesn't author.
	if !strings.Contains(got, "NEVER fabricate") {
		t.Errorf("anti-pattern: never-fabricate guard missing")
	}
	// JSON-only contract.
	if !strings.Contains(got, "NEVER write commentary outside the JSON") {
		t.Errorf("anti-pattern: JSON-only guard missing")
	}
	// Input-unparseable fail-loud envelope.
	if !strings.Contains(got, "input_unparseable") {
		t.Errorf("fail-loud guard for input_unparseable missing")
	}
}

// -----------------------------------------------------------------------------
// Pure function — deterministic per ADR-141 D2 transparency
// -----------------------------------------------------------------------------

func TestComposeCriticInstruction_Deterministic(t *testing.T) {
	t.Parallel()
	ctx := newMCQCriticCtx()
	a := ComposeCriticInstruction(CriticStep(), ctx)
	b := ComposeCriticInstruction(CriticStep(), ctx)
	if a != b {
		t.Errorf("ComposeCriticInstruction should be deterministic; got drift between runs")
	}
}

// -----------------------------------------------------------------------------
// resolveCriticTaskAndOutput — direct table-driven
// -----------------------------------------------------------------------------

func TestResolveCriticTaskAndOutput_Table(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		qt   QuestionType
		want string // a string that must appear in the resolved TaskBlock
	}{
		{name: "mcq", qt: QuestionTypeMCQ, want: "MCQ candidate from the qgen_question generator"},
		{name: "oe", qt: QuestionTypeOE, want: "OE candidate from the qgen_question generator"},
		{name: "unsupported", qt: QuestionType("video"), want: "invalid question_type"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx := newMCQCriticCtx()
			ctx.QuestionType = tc.qt
			task, _ := resolveCriticTaskAndOutput(ctx)
			if !strings.Contains(task, tc.want) {
				t.Errorf("task missing %q for qt=%s; got\n%s", tc.want, tc.qt, task)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// HARD REJECTION CRITERIA (tightened 2026-05-17 — see test_*_critique_
// qualitative_only integration smokes). These tests assert that the TASK
// blocks carry the objective gates the live integration tests rely on.
// -----------------------------------------------------------------------------

// TestMCQTaskBlock_HardRejectionCriteria_OptionCount asserts the MCQ task
// block explicitly demands 3-5 options, the gate the deliberately-weak
// 2-option MCQ smoke trips (test_mcq_critique_qualitative_only).
func TestMCQTaskBlock_HardRejectionCriteria_OptionCount(t *testing.T) {
	t.Parallel()
	task := taskCritiqueMCQ()
	// Hard criterion H1 — option count.
	if !strings.Contains(task, "HARD REJECTION CRITERIA") {
		t.Fatalf("MCQ task block missing HARD REJECTION CRITERIA banner; got:\n%s", task)
	}
	for _, want := range []string{
		"3-5 options",
		"H1",
		"only 2 options",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("MCQ task block missing option-count language %q", want)
		}
	}
}

// TestMCQTaskBlock_HardRejectionCriteria_Explainer asserts the explainer
// gate is explicit. Tautological explainers (the smoke's "It is a process.")
// must be a REJECT.
func TestMCQTaskBlock_HardRejectionCriteria_Explainer(t *testing.T) {
	t.Parallel()
	task := taskCritiqueMCQ()
	for _, want := range []string{
		"H2",
		"non-empty `explainer`",
		"Tautological",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("MCQ task block missing explainer-gate language %q", want)
		}
	}
}

// TestMCQTaskBlock_NoStemWordCountRule guards the user directive (2026-06-01):
// there is NO minimum stem word-count gate. A concise, clear question
// ("What gas do plants absorb during photosynthesis?") is a valid MCQ stem;
// stem quality is judged qualitatively (clarity), never by length. Regression
// guard for the removed H3 stem-length rule.
func TestMCQTaskBlock_NoStemWordCountRule(t *testing.T) {
	t.Parallel()
	task := taskCritiqueMCQ()
	for _, banned := range []string{
		"at least 10 words",
		"10 words",
		"Stem substance",
		"Trivially short",
	} {
		if strings.Contains(task, banned) {
			t.Errorf("MCQ task block still imposes a stem word-count rule (%q); removed per user directive", banned)
		}
	}
}

// TestMCQTaskBlock_HardRejectionCriteria_ObviousAnswer asserts the
// obvious-correct-answer pattern is called out — the smoke's "A process" vs
// "Something else" pair trips this.
func TestMCQTaskBlock_HardRejectionCriteria_ObviousAnswer(t *testing.T) {
	t.Parallel()
	task := taskCritiqueMCQ()
	for _, want := range []string{
		"H3", // renumbered from H4 after the stem word-count rule was removed
		"Obvious-correct-answer",
		"Something else",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("MCQ task block missing obvious-answer-pattern language %q", want)
		}
	}
}

// TestOETaskBlock_HardRejectionCriteria_ModelAnswerWordCount asserts the OE
// model_answer word-count gate is explicit (≥40 words) — the smoke's 12-word
// model_answer must trip H1.
func TestOETaskBlock_HardRejectionCriteria_ModelAnswerWordCount(t *testing.T) {
	t.Parallel()
	task := taskCritiqueOE()
	if !strings.Contains(task, "HARD REJECTION CRITERIA") {
		t.Fatalf("OE task block missing HARD REJECTION CRITERIA banner; got:\n%s", task)
	}
	for _, want := range []string{
		"H1",
		"at least 40 words",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("OE task block missing model_answer-word-count language %q", want)
		}
	}
}

// TestOETaskBlock_HardRejectionCriteria_RubricCount asserts the OE rubric
// criterion-count gate is explicit (≥3 criteria) — the smoke's 1-item rubric
// must trip H2.
func TestOETaskBlock_HardRejectionCriteria_RubricCount(t *testing.T) {
	t.Parallel()
	task := taskCritiqueOE()
	for _, want := range []string{
		"H2",
		"at least 3 criteria",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("OE task block missing rubric-count language %q", want)
		}
	}
}

// TestOETaskBlock_HardRejectionCriteria_WeightSumTolerance asserts the OE
// weight-sum tolerance gate accepts both 1.0±1% (fractional) and 100±1%
// (percentage) conventions — the smoke's single-weight 0.5 trips H3 under
// the fractional convention.
func TestOETaskBlock_HardRejectionCriteria_WeightSumTolerance(t *testing.T) {
	t.Parallel()
	task := taskCritiqueOE()
	for _, want := range []string{
		"H3",
		"1.0 ± 1%",
		"100 ± 1%",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("OE task block missing weight-sum-tolerance language %q", want)
		}
	}
}

// TestOETaskBlock_HardRejectionCriteria_GraderTier asserts the grader_tier
// presence + enum gate is explicit — the smoke's missing-grader_tier OE must
// trip H4.
func TestOETaskBlock_HardRejectionCriteria_GraderTier(t *testing.T) {
	t.Parallel()
	task := taskCritiqueOE()
	for _, want := range []string{
		"H4",
		"`grader_tier` MUST be present",
		"T1\" or \"T2",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("OE task block missing grader_tier-gate language %q", want)
		}
	}
}

// TestComposeCriticInstruction_ExamplesShowHardCriteriaRejection asserts the
// [EXAMPLES] block carries concrete weak-input rejection patterns that
// mirror the smoke fixtures (so the LLM short-circuits objectively).
func TestComposeCriticInstruction_ExamplesShowHardCriteriaRejection(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	for _, want := range []string{
		"MCQ HARD-REJECTION",
		"only 2 options provided",
		"OE HARD-REJECTION",
		"model_answer has 12 words",
		"grader_tier missing",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("composed prompt missing hard-rejection example %q", want)
		}
	}
}

// TestComposeCriticInstruction_CritiqueNotesCitesSpecificGap asserts the TASK
// block requires critique_notes to cite the specific gap (hard-criterion ID
// + observed value) so the regenerator gets actionable feedback.
func TestComposeCriticInstruction_CritiqueNotesCitesSpecificGap(t *testing.T) {
	t.Parallel()
	mcqTask := taskCritiqueMCQ()
	oeTask := taskCritiqueOE()
	// MCQ — cite hard-criterion ID OR option_id.
	if !strings.Contains(mcqTask, "critique_notes MUST cite the SPECIFIC gap") {
		t.Errorf("MCQ task block should require critique_notes to cite specific gap")
	}
	// OE — cite hard-criterion ID + observed value.
	if !strings.Contains(oeTask, "critique_notes MUST cite the SPECIFIC gap") {
		t.Errorf("OE task block should require critique_notes to cite specific gap")
	}
}

// -----------------------------------------------------------------------------
// Output JSON schema — examples still parse as valid JSON after tightening
// -----------------------------------------------------------------------------

// TestOutputCritiqueMCQ_SchemaIsParseableShape asserts the example
// {accepted, critique_notes, suggested_revisions} envelope in outputCritiqueMCQ
// remains a parseable JSON shape — guards against accidental schema drift
// during prompt tightening.
func TestOutputCritiqueMCQ_SchemaIsParseableShape(t *testing.T) {
	t.Parallel()
	out := outputCritiqueMCQ()
	for _, want := range []string{
		"\"accepted\": bool",
		"\"critique_notes\": string",
		"\"suggested_revisions\": [string, ...]",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("outputCritiqueMCQ missing JSON schema fragment %q", want)
		}
	}
}

// TestOutputCritiqueOE_SchemaIsParseableShape — same shape gate for OE.
func TestOutputCritiqueOE_SchemaIsParseableShape(t *testing.T) {
	t.Parallel()
	out := outputCritiqueOE()
	for _, want := range []string{
		"\"accepted\": bool",
		"\"critique_notes\": string",
		"\"suggested_revisions\": [string, ...]",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("outputCritiqueOE missing JSON schema fragment %q", want)
		}
	}
}

// TestOutputCritiqueMCQ_NonEmptyNotesRequiredOnReject asserts the contract
// that critique_notes MUST be non-empty when accepted=false is still
// surfaced after prompt-tightening.
func TestOutputCritiqueMCQ_NonEmptyNotesRequiredOnReject(t *testing.T) {
	t.Parallel()
	for _, out := range []string{outputCritiqueMCQ(), outputCritiqueOE()} {
		if !strings.Contains(out, "accepted=false: critique_notes MUST be non-empty") {
			t.Errorf("output block missing non-empty-notes-on-reject contract; got:\n%s", out)
		}
	}
}

// TestOutputCritique_AcceptRationaleRequired asserts that on ACCEPT the critic
// must emit a concise critique_notes rationale (not empty). This is the IMDA D2
// audit record an auditor reads in O+ Decision-Traces — a clean
// PASS used to emit critique_notes="" so the row only showed the generic
// "critic accepted" fallback. Both MCQ + OE output blocks carry the contract.
func TestOutputCritique_AcceptRationaleRequired(t *testing.T) {
	t.Parallel()
	for _, out := range []string{outputCritiqueMCQ(), outputCritiqueOE()} {
		if !strings.Contains(out, "accepted=true: critique_notes MUST be a concise") {
			t.Errorf("output block missing accept-rationale contract; got:\n%s", out)
		}
		// The stale "MAY be empty" accept license must be gone.
		if strings.Contains(out, "accepted=true: critique_notes MAY be empty") {
			t.Errorf("output block still licenses empty accept notes; got:\n%s", out)
		}
	}
}

// TestComposeCriticInstruction_AcceptExamplesCarryRationale asserts the composed
// prompt's ACCEPT examples now model a non-empty rationale (not critique_notes
// "") so the LLM produces a real PASS rationale.
func TestComposeCriticInstruction_AcceptExamplesCarryRationale(t *testing.T) {
	t.Parallel()
	got := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())
	// The empty-notes accept example shapes must be gone from the prompt.
	if strings.Contains(got, "\"accepted\": true, \"critique_notes\": \"\"") {
		t.Errorf("accept example still emits empty critique_notes; got:\n%s", got)
	}
}

// -----------------------------------------------------------------------------
// ADR-197 M-B — operator SAFE-block overrides (read side), critic composer.
// -----------------------------------------------------------------------------

func TestComposeCritic_overridesNil_behaviourNeutral(t *testing.T) {
	t.Parallel()
	// With no SAFE-block override the composed critic prompt is identical whether
	// Overrides is nil, empty, carries only non-SAFE keys, or blank SAFE values.
	base := ComposeCriticInstruction(CriticStep(), newMCQCriticCtx())

	empty := newMCQCriticCtx()
	empty.Overrides = map[string]string{}
	if got := ComposeCriticInstruction(CriticStep(), empty); got != base {
		t.Errorf("empty overrides changed the critic prompt:\n%s", firstDiff(base, got))
	}

	nonSafe := newMCQCriticCtx()
	nonSafe.Overrides = map[string]string{"output": "HACKED", "candidate": "HACKED", "audience": "HACKED"}
	if got := ComposeCriticInstruction(CriticStep(), nonSafe); got != base {
		t.Errorf("non-SAFE override keys leaked into the critic prompt:\n%s", firstDiff(base, got))
	}

	blank := newMCQCriticCtx()
	blank.Overrides = map[string]string{"role": "", "task": "", "examples": ""}
	if got := ComposeCriticInstruction(CriticStep(), blank); got != base {
		t.Errorf("blank SAFE overrides changed the critic prompt:\n%s", firstDiff(base, got))
	}
}

func TestComposeCritic_overridesSubstituteSafeBlocks(t *testing.T) {
	t.Parallel()
	step := CriticStep()
	base := ComposeCriticInstruction(step, newMCQCriticCtx())

	ctx := newMCQCriticCtx()
	ctx.Overrides = map[string]string{
		"role":     "OVERRIDE_ROLE_SENTINEL",
		"task":     "OVERRIDE_TASK_SENTINEL",
		"examples": "OVERRIDE_EXAMPLES_SENTINEL",
	}
	got := ComposeCriticInstruction(step, ctx)

	if !strings.Contains(got, "## [ROLE]\nOVERRIDE_ROLE_SENTINEL\n\n") {
		t.Errorf("role override not applied at [ROLE]; got:\n%s", got)
	}
	if !strings.Contains(got, "## [TASK]\nOVERRIDE_TASK_SENTINEL\n\n") {
		t.Errorf("task override not applied at [TASK]; got:\n%s", got)
	}
	if !strings.Contains(got, "## [EXAMPLES]\nOVERRIDE_EXAMPLES_SENTINEL") {
		t.Errorf("examples override not applied at [EXAMPLES]; got:\n%s", got)
	}
	if strings.Contains(got, step.RoleBlock) {
		t.Errorf("embedded critic role block still present after override")
	}

	// The candidate JSON, the [EXPECTED OUTPUT] JSON-contract block, and the
	// "NEVER fabricate" safety preamble are NEVER overridable — the tail from the
	// [EXPECTED OUTPUT] header must be byte-identical across the override.
	const marker = "## [EXPECTED OUTPUT]\n"
	if base[strings.Index(base, marker):] != got[strings.Index(got, marker):] {
		t.Errorf("[EXPECTED OUTPUT] block changed under overrides — it must be immutable")
	}
	// The candidate is still present verbatim (it lives in its own non-overridable block).
	if !strings.Contains(got, ctx.CandidateJSON) {
		t.Errorf("candidate JSON dropped under overrides")
	}
}

func TestBuildCriticTaskContextFromState_readsOverridesAndVersion(t *testing.T) {
	t.Parallel()
	st := &fakeState{data: map[string]any{
		"question_type":           "mcq",
		"input_payload":           `{"stem":"x","question_type":"mcq"}`,
		"prompt_overrides_json":   `{"role":"R"}`,
		"resolved_prompt_version": "v9",
	}}
	tc, err := BuildCriticTaskContextFromState(st)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if tc.Overrides["role"] != "R" {
		t.Errorf("overrides not read: %#v", tc.Overrides)
	}
	if tc.ResolvedPromptVersion != "v9" {
		t.Errorf("resolved_prompt_version=%q want v9", tc.ResolvedPromptVersion)
	}
}
