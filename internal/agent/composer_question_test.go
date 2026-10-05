package agent

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// composer_question.go is the 3-agent qgen_question composer per ADR-153.
// It mirrors composer.go but emits only 3 steps and supports the Intent +
// QuestionType runtime branching that the single-Q AI-assist path needs:
//
//   1. content_assurance — pre-gen safety + multimodal-capable file parsing
//      (gemini-2.5-pro, parse_document tool registered)
//   2. qgen_generation   — 4 prompt templates (new_mcq / new_oe / fill_mcq /
//      fill_oe) selected via Intent + QuestionType
//   3. qgen_evaluation   — LLM-as-judge, same 3 axes as the 6-agent crew
//
// Tests cover:
//   - 3 steps in canonical order
//   - canonical names (qgen_question_assurance / qgen_question_generation /
//     qgen_question_evaluation) so OTLP `chora.qgen.step` traces stay
//     parseable and recovery-stable (D6 P1).
//   - deterministic composition (D2 transparency per ADR-141)
//   - context fields surface through into [CONTEXT] block
//   - Intent + QuestionType runtime branching produces 4 distinct
//     generation prompts (new_mcq / new_oe / fill_mcq / fill_oe).
//   - assurance prompt carries the parse_document tool advertisement (so
//     the LLM knows the tool exists, even when not invoked).
//   - assurance prompt carries the new_question vs model_answer_fill
//     intent cue.
//   - Mandatory span attributes ARE re-exposed via the same shared
//     MandatorySpanAttributes() (single crew-wide contract — D6 P4).

func ctxNewMCQ() TaskContextQuestion {
	return TaskContextQuestion{
		TenantID:     "01957c8c-0000-7000-8888-000088880000",
		BatchID:      "01957c8c-9999-7000-7777-9999aaaa9999",
		AuthorGCID:   "01957c8c-0000-7000-9999-000099990000",
		Intent:       IntentNewQuestion,
		QuestionType: QuestionTypeMCQ,
		Prompt:       "Generate an MCQ on Scrum sprints",
	}
}

func ctxNewOE() TaskContextQuestion {
	return TaskContextQuestion{
		TenantID:     "tenant-x",
		BatchID:      "batch-x",
		AuthorGCID:   "gcid-x",
		Intent:       IntentNewQuestion,
		QuestionType: QuestionTypeOE,
		Prompt:       "Explain why story points are preferred over hours",
	}
}

func ctxFillMCQ() TaskContextQuestion {
	return TaskContextQuestion{
		TenantID:     "tenant-x",
		BatchID:      "batch-x",
		AuthorGCID:   "gcid-x",
		Intent:       IntentModelAnswerFill,
		QuestionType: QuestionTypeMCQ,
		Prompt:       "Which Scrum role owns the backlog?",
		ExistingQuestion: &ExistingQuestion{
			Stem: "Which Scrum role owns the backlog?",
			MCQOptions: []MCQOption{
				{OptionID: "opt-1", Label: "Scrum Master", IsCorrect: false},
				{OptionID: "opt-2", Label: "Product Owner", IsCorrect: true},
				{OptionID: "opt-3", Label: "Developer", IsCorrect: false},
				{OptionID: "opt-4", Label: "Stakeholder", IsCorrect: false},
			},
		},
	}
}

func ctxFillOE() TaskContextQuestion {
	return TaskContextQuestion{
		TenantID:     "tenant-x",
		BatchID:      "batch-x",
		AuthorGCID:   "gcid-x",
		Intent:       IntentModelAnswerFill,
		QuestionType: QuestionTypeOE,
		Prompt:       "Explain story points",
		ExistingQuestion: &ExistingQuestion{
			Stem: "Explain story points",
			OERubric: []OERubricCriterion{
				{Criterion: "Defines relativity", Weight: 0.5},
				{Criterion: "Captures complexity", Weight: 0.5},
			},
		},
	}
}

// -----------------------------------------------------------------------------
// Step inventory + canonical names
// -----------------------------------------------------------------------------

func TestAllStepsQuestion_returnsGenerationOnly(t *testing.T) {
	// Trimmed 2026-06-01 (user directive, latency): assurance + evaluation
	// dropped from the single-Q pipeline. Model Armor guardrail_pre covers
	// pre-gen safety; the now-working qgen_critic does the quality judging.
	// StepAssurance3 + StepEvaluation3 remain defined (reusable constructors)
	// but are no longer wired here.
	steps := AllStepsQuestion()
	if len(steps) != 1 {
		t.Fatalf("AllStepsQuestion() must return 1 step (generation-only after the trim); got %d", len(steps))
	}
	if steps[0].Name != "qgen_question_generation" {
		t.Errorf("step 0: want %q; got %q", "qgen_question_generation", steps[0].Name)
	}
}

func TestStepQuestionConstructors_haveCanonicalNames(t *testing.T) {
	pairs := []struct{ got, want string }{
		{StepAssurance3().Name, "qgen_question_assurance"},
		{StepGeneration3().Name, "qgen_question_generation"},
		{StepEvaluation3().Name, "qgen_question_evaluation"},
	}
	for _, p := range pairs {
		if p.got != p.want {
			t.Errorf("step name: got %q want %q", p.got, p.want)
		}
	}
}

// -----------------------------------------------------------------------------
// CREATE-block composition (6 canonical blocks; mirrors §composer_test.go)
// -----------------------------------------------------------------------------

func TestComposeQGenQuestionInstruction_emitsSixCreateBlocksInOrder(t *testing.T) {
	contexts := []TaskContextQuestion{ctxNewMCQ(), ctxNewOE(), ctxFillMCQ(), ctxFillOE()}
	for _, c := range contexts {
		for _, step := range AllStepsQuestion() {
			got := ComposeQGenQuestionInstruction(step, c)
			blocks := []string{
				"[CONTEXT]",
				"[ROLE]",
				"[EXAMPLES]",
				"[AUDIENCE]",
				"[TASK]",
				"[EXPECTED OUTPUT]",
			}
			lastIdx := -1
			for _, b := range blocks {
				i := strings.Index(got, b)
				if i < 0 {
					t.Errorf("intent=%s qt=%s step=%s: CREATE block %q missing\n--- prompt:\n%s",
						c.Intent, c.QuestionType, step.Name, b, got)
					continue
				}
				if i <= lastIdx {
					t.Errorf("intent=%s qt=%s step=%s: CREATE blocks out of order; %q at %d after %d",
						c.Intent, c.QuestionType, step.Name, b, i, lastIdx)
				}
				lastIdx = i
			}
		}
	}
}

func TestComposeQGenQuestionInstruction_isDeterministic(t *testing.T) {
	for _, c := range []TaskContextQuestion{ctxNewMCQ(), ctxNewOE(), ctxFillMCQ(), ctxFillOE()} {
		for _, step := range AllStepsQuestion() {
			a := ComposeQGenQuestionInstruction(step, c)
			b := ComposeQGenQuestionInstruction(step, c)
			if a != b {
				t.Errorf("intent=%s qt=%s step=%s: ComposeQGenQuestionInstruction not deterministic (ADR-141 D2 break)",
					c.Intent, c.QuestionType, step.Name)
			}
		}
	}
}

func TestComposeQGenQuestionInstruction_distinguishesSteps(t *testing.T) {
	c := ctxNewMCQ()
	prompts := make(map[string]string, 3)
	for _, step := range AllStepsQuestion() {
		prompts[step.Name] = ComposeQGenQuestionInstruction(step, c)
	}
	for n1, p1 := range prompts {
		for n2, p2 := range prompts {
			if n1 != n2 && p1 == p2 {
				t.Errorf("steps %s and %s produced identical instructions", n1, n2)
			}
		}
	}
}

func TestComposeQGenQuestionInstruction_contextFieldsSurface(t *testing.T) {
	c := TaskContextQuestion{
		TenantID:     "tenant-xyz",
		BatchID:      "batch-abc",
		AuthorGCID:   "gcid-phyllis",
		Intent:       IntentNewQuestion,
		QuestionType: QuestionTypeMCQ,
		Prompt:       "Generate one MCQ on graph algorithms",
	}
	got := ComposeQGenQuestionInstruction(StepAssurance3(), c)
	for _, want := range []string{"tenant-xyz", "batch-abc", "gcid-phyllis", "new_question", "mcq"} {
		if !strings.Contains(got, want) {
			t.Errorf("CONTEXT block missing %q; prompt:\n%s", want, got)
		}
	}
}

// Authoring-metadata wire-through (2026-06-03): the single-Q generator must
// condition on the author's Subject / Cognitive Level / Difficulty. The
// executor stamps subject_hint / cognitive_level_hint / difficulty_hint;
// BuildTaskContextFromState reads them and ComposeQGenQuestionInstruction
// surfaces them in [CONTEXT] (mirrors critic.go metadataHints format).

func TestBuildTaskContextFromState_readsMetadataHints(t *testing.T) {
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":            "tenant-x",
		"author_gcid":          "gcid-x",
		"intent":               "new_question",
		"question_type":        "mcq",
		"input_payload":        "Newton's first law",
		"subject_hint":         "Physics — Newtonian mechanics",
		"cognitive_level_hint": "synthesis",
		"difficulty_hint":      "advanced",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.SubjectHint != "Physics — Newtonian mechanics" {
		t.Errorf("SubjectHint = %q; want %q", tc.SubjectHint, "Physics — Newtonian mechanics")
	}
	if tc.CognitiveLevelHint != "synthesis" {
		t.Errorf("CognitiveLevelHint = %q; want %q", tc.CognitiveLevelHint, "synthesis")
	}
	if tc.DifficultyHint != "advanced" {
		t.Errorf("DifficultyHint = %q; want %q", tc.DifficultyHint, "advanced")
	}
}

func TestComposeQGenQuestionInstruction_metadataHintsSurface(t *testing.T) {
	c := TaskContextQuestion{
		TenantID:           "tenant-xyz",
		AuthorGCID:         "gcid-phyllis",
		Intent:             IntentNewQuestion,
		QuestionType:       QuestionTypeMCQ,
		Prompt:             "Generate one MCQ on Newton's first law",
		SubjectHint:        "Physics — Newtonian mechanics",
		CognitiveLevelHint: "synthesis",
		DifficultyHint:     "advanced",
	}
	got := ComposeQGenQuestionInstruction(StepAssurance3(), c)
	for _, want := range []string{
		"subject=Physics — Newtonian mechanics",
		"cognitive_level=synthesis",
		"difficulty=advanced",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("CONTEXT block missing hint %q; prompt:\n%s", want, got)
		}
	}
}

func TestComposeQGenQuestionInstruction_metadataHintsOmittedWhenEmpty(t *testing.T) {
	c := TaskContextQuestion{
		TenantID:     "tenant-xyz",
		AuthorGCID:   "gcid-phyllis",
		Intent:       IntentNewQuestion,
		QuestionType: QuestionTypeMCQ,
		Prompt:       "Generate one MCQ",
	}
	got := ComposeQGenQuestionInstruction(StepAssurance3(), c)
	for _, notWant := range []string{"subject=", "cognitive_level=", "difficulty=", "Author hints"} {
		if strings.Contains(got, notWant) {
			t.Errorf("CONTEXT block should omit hint token %q when unset; prompt:\n%s", notWant, got)
		}
	}
}

// -----------------------------------------------------------------------------
// Assurance prompt — multimodal cue + parse_document tool advertisement
// -----------------------------------------------------------------------------

func TestStepAssurance3_advertisesParseDocumentTool(t *testing.T) {
	// Per ADR-153 the assurance agent registers a parse_document tool. The
	// prompt MUST advertise the tool so the LLM knows it can be invoked
	// when a file is attached. For single-Q text-only the tool is dormant.
	got := ComposeQGenQuestionInstruction(StepAssurance3(), ctxNewMCQ())
	for _, cue := range []string{
		"parse_document",
		"multimodal",
	} {
		if !strings.Contains(got, cue) {
			t.Errorf("Assurance prompt missing %q advertisement", cue)
		}
	}
}

func TestStepAssurance3_carriesIntentCue(t *testing.T) {
	// Assurance must read the intent so it can apply the right downstream
	// expectations (new_question vs model_answer_fill).
	for _, c := range []TaskContextQuestion{ctxNewMCQ(), ctxFillOE()} {
		got := ComposeQGenQuestionInstruction(StepAssurance3(), c)
		if !strings.Contains(got, string(c.Intent)) {
			t.Errorf("intent=%s: Assurance prompt missing intent cue %q", c.Intent, c.Intent)
		}
	}
}

func TestStepAssurance3_emitsExpectedJSONShape(t *testing.T) {
	// Assurance OutputBlock declares the {valid, reason, parsed_content,
	// redacted_input} verdict shape per ADR-153.
	got := ComposeQGenQuestionInstruction(StepAssurance3(), ctxNewMCQ())
	for _, key := range []string{"valid", "reason", "parsed_content", "redacted_input"} {
		if !strings.Contains(got, key) {
			t.Errorf("Assurance OutputBlock missing %q field", key)
		}
	}
}

// -----------------------------------------------------------------------------
// Generation prompt — 4-template runtime branching
// -----------------------------------------------------------------------------

func TestStepGeneration3_branchesOn4Combinations(t *testing.T) {
	// 4 distinct generation prompts must result from the 2 Intent × 2
	// QuestionType matrix. We compare them pairwise; any two equal = drift.
	prompts := map[string]string{
		"new_mcq":  ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQ()),
		"new_oe":   ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewOE()),
		"fill_mcq": ComposeQGenQuestionInstruction(StepGeneration3(), ctxFillMCQ()),
		"fill_oe":  ComposeQGenQuestionInstruction(StepGeneration3(), ctxFillOE()),
	}
	keys := []string{"new_mcq", "new_oe", "fill_mcq", "fill_oe"}
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			if prompts[keys[i]] == prompts[keys[j]] {
				t.Errorf("Generation prompts for %s and %s are identical; runtime branching broken",
					keys[i], keys[j])
			}
		}
	}
}

func TestStepGeneration3_newMCQ_advertisesNewQuestionDirective(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQ())
	if !strings.Contains(strings.ToLower(got), "draft a new mcq") &&
		!strings.Contains(strings.ToLower(got), "produce a new mcq") &&
		!strings.Contains(strings.ToLower(got), "generate a new mcq") {
		t.Errorf("new_mcq generation prompt does not direct the LLM to draft a new MCQ\n%s", got)
	}
	if !strings.Contains(strings.ToLower(got), "distractor") {
		t.Errorf("new_mcq generation prompt does not mention distractors")
	}
}

func TestStepGeneration3_newOE_directsModelAnswerOutput(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewOE())
	low := strings.ToLower(got)
	if !strings.Contains(low, "open-ended") && !strings.Contains(low, "open ended") {
		t.Errorf("new_oe prompt does not declare open-ended question type")
	}
	if !strings.Contains(low, "model_answer") && !strings.Contains(low, "model answer") {
		t.Errorf("new_oe prompt does not request a model answer")
	}
}

func TestStepGeneration3_fillMCQ_preservesUserOptions(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxFillMCQ())
	// The author's existing options MUST surface into the prompt so the
	// LLM fills per-option explainers against them, NOT redrafts the stem.
	for _, label := range []string{"Product Owner", "Scrum Master", "Developer", "Stakeholder"} {
		if !strings.Contains(got, label) {
			t.Errorf("fill_mcq prompt missing existing option label %q", label)
		}
	}
	if !strings.Contains(strings.ToLower(got), "explainer") {
		t.Errorf("fill_mcq prompt does not request per-option explainers")
	}
}

func TestStepGeneration3_fillOE_surfacesRubric(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxFillOE())
	for _, criterion := range []string{"Defines relativity", "Captures complexity"} {
		if !strings.Contains(got, criterion) {
			t.Errorf("fill_oe prompt missing rubric criterion %q", criterion)
		}
	}
	if !strings.Contains(strings.ToLower(got), "model answer") && !strings.Contains(strings.ToLower(got), "model_answer") {
		t.Errorf("fill_oe prompt does not request a model answer")
	}
}

// -----------------------------------------------------------------------------
// Evaluation prompt — LLM-as-judge, same 3 axes
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// OE template — nested oe_payload + rubric + grader_tier (Acceptance Gate #2)
// -----------------------------------------------------------------------------
//
// Per docs/m13/ack-oe-ai-assist-plan-2026-05-17.md §5 + the canonical OpenAPI
// `OEPayload` schema in chora-contracts/openapi/creation-questions.yaml
// §1100-1121, the OE candidate must nest answer data inside `oe_payload`
// with model_answer + rubric (≥3 entries, weights sum to 1.0) + grader_tier
// ("T1"|"T2"). The prior FLAT shape was a divergence surfaced by the OE
// smoke test.

func TestStepGeneration3_newOE_emitsNestedOEPayload(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewOE())
	// Expected-output block must declare the nested oe_payload + the
	// three load-bearing fields plus rubric per-entry shape.
	for _, want := range []string{
		"oe_payload",
		"rubric",
		"grader_tier",
		"criterion_id",
		"description",
		"weight",
		"T1",
		"T2",
		"40 words",
		"sum to 1.0",
		"≥3", // "≥3" — rubric size invariant.
	} {
		if !strings.Contains(got, want) {
			t.Errorf("new_oe generation prompt missing %q\n--- prompt:\n%s", want, got)
		}
	}
}

func TestStepGeneration3_newOE_exampleIsParseableJSON(t *testing.T) {
	// The [EXPECTED OUTPUT] example must contain a syntactically valid
	// JSON object with the canonical oe_payload nesting. We extract the
	// {"candidate": ...} object from the example block and json.Unmarshal it.
	prompt := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewOE())
	// Find the literal "{\"candidate\": {\"stem\": \"Explain how" — the
	// example anchor — then walk braces to extract a balanced object.
	anchor := `{"candidate": {"stem": "Explain how`
	idx := strings.Index(prompt, anchor)
	if idx < 0 {
		t.Fatalf("new_oe expected-output example missing canonical anchor.\n--- prompt:\n%s", prompt)
	}
	depth := 0
	start := idx
	end := -1
	for i := start; i < len(prompt); i++ {
		switch prompt[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i + 1 // loop exit handled by `if end > 0 { break }` below
			}
		}
		if end > 0 {
			break
		}
	}
	if end < 0 {
		t.Fatalf("new_oe example JSON braces unbalanced.\n--- prompt:\n%s", prompt)
	}
	raw := prompt[start:end]
	var parsed struct {
		Candidate struct {
			Stem         string `json:"stem"`
			QuestionType string `json:"question_type"`
			Intent       string `json:"intent"`
			OEPayload    struct {
				ModelAnswer string `json:"model_answer"`
				Rubric      []struct {
					CriterionID string  `json:"criterion_id"`
					Title       string  `json:"title"`
					Description string  `json:"description"`
					Weight      float64 `json:"weight"`
				} `json:"rubric"`
				GraderTier string `json:"grader_tier"`
			} `json:"oe_payload"`
		} `json:"candidate"`
	}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		t.Fatalf("new_oe example JSON failed to parse: %v\n--- raw:\n%s", err, raw)
	}
	if parsed.Candidate.QuestionType != "oe" {
		t.Errorf("example question_type = %q; want \"oe\"", parsed.Candidate.QuestionType)
	}
	if parsed.Candidate.Intent != "new_question" {
		t.Errorf("example intent = %q; want \"new_question\"", parsed.Candidate.Intent)
	}
	if parsed.Candidate.OEPayload.ModelAnswer == "" {
		t.Errorf("example model_answer is empty")
	}
	if wc := len(strings.Fields(parsed.Candidate.OEPayload.ModelAnswer)); wc < 40 {
		t.Errorf("example model_answer word count %d < 40", wc)
	}
	if got := len(parsed.Candidate.OEPayload.Rubric); got < 3 {
		t.Errorf("example rubric has %d entries; need ≥3", got)
	}
	var sum float64
	for _, c := range parsed.Candidate.OEPayload.Rubric {
		if c.CriterionID == "" || c.Title == "" || c.Description == "" {
			t.Errorf("example rubric entry missing field: %+v", c)
		}
		if c.Weight < 0 || c.Weight > 1 {
			t.Errorf("example rubric weight %v out of [0,1]", c.Weight)
		}
		sum += c.Weight
	}
	if delta := sum - 1.0; delta < -0.01 || delta > 0.01 {
		t.Errorf("example rubric weights sum = %v; want 1.0 ±0.01", sum)
	}
	if parsed.Candidate.OEPayload.GraderTier != "T1" && parsed.Candidate.OEPayload.GraderTier != "T2" {
		t.Errorf("example grader_tier = %q; want T1 or T2", parsed.Candidate.OEPayload.GraderTier)
	}
}

func TestStepGeneration3_fillOE_emitsNestedOEPayloadAndPreservesRubric(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxFillOE())
	// Nested oe_payload + same fields as new_oe — but the task body must
	// also keep its "preserve author stem + rubric" invariant.
	for _, want := range []string{
		"oe_payload",
		"rubric",
		"grader_tier",
		"criterion_id",
		"T1",
		"T2",
		"40 words",
		"round-trip",
		"NEVER alter the stem",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("fill_oe generation prompt missing %q\n--- prompt:\n%s", want, got)
		}
	}
	// Author rubric criteria still surface (preservation invariant).
	for _, criterion := range []string{"Defines relativity", "Captures complexity"} {
		if !strings.Contains(got, criterion) {
			t.Errorf("fill_oe prompt dropped author rubric criterion %q", criterion)
		}
	}
}

func TestStepGeneration3_newOE_doesNotEmitOldFlatShape(t *testing.T) {
	// Regression guard — the old flat outputBlock was a single line
	// `{"candidate": {"stem": ..., "model_answer": ..., "intent": ...,
	// "question_type": ...}}`. After the bump, model_answer MUST NOT
	// appear at the candidate top-level (only inside oe_payload). We
	// assert by checking the candidate-level (top-level under "candidate":
	// {) keys do not include model_answer outside the nested wrapper.
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewOE())
	// In the OUTPUT block the schema line is:
	//   {"candidate": {"stem": string, "question_type": "oe",
	//   "intent": "new_question", "oe_payload": {"model_answer": ...}}}.
	// A flat shape would have `"model_answer": string,` directly
	// after the candidate-level "stem": string. Detect by looking for
	// the substring `"stem": string, "model_answer"` (flat) and ensure
	// it is absent.
	if strings.Contains(got, "\"stem\": string, \"model_answer\"") {
		t.Errorf("new_oe outputBlock still emits FLAT shape (model_answer at top level)")
	}
}

func TestStepEvaluation3_declaresThreeAxes(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepEvaluation3(), ctxNewMCQ())
	low := strings.ToLower(got)
	for _, axis := range []string{"factuality", "clarity", "difficulty"} {
		if !strings.Contains(low, axis) {
			t.Errorf("Evaluation prompt missing axis %q", axis)
		}
	}
	if !strings.Contains(low, "composite") {
		t.Errorf("Evaluation prompt missing composite score")
	}
}

// -----------------------------------------------------------------------------
// BuildTaskContextFromState — M14.2 runtime per-turn injection
// -----------------------------------------------------------------------------

// fakeState is an in-memory implementation of stateGetter the tests use
// to simulate session.ReadonlyState without booting the full ADK
// runtime. Mirrors the pattern used in
// instancedispatch_test.go / manaplugin_test.go.
type fakeState struct {
	data map[string]any
}

func newFakeState(data map[string]any) *fakeState {
	if data == nil {
		data = map[string]any{}
	}
	return &fakeState{data: data}
}

func (f *fakeState) Get(k string) (any, error) {
	v, ok := f.data[k]
	if !ok {
		return nil, fmt.Errorf("key %q not found", k)
	}
	return v, nil
}

func TestBuildTaskContextFromState_defaultsToMCQNewQuestionOnMissingKeys(t *testing.T) {
	// Missing intent + question_type → defaults preserve the pre-M14.2
	// behaviour (so the worst-case live engine still emits a working
	// MCQ new-question prompt).
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":   "tenant-x",
		"author_gcid": "gcid-x",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.Intent != IntentNewQuestion {
		t.Errorf("intent = %q; want %q (default)", tc.Intent, IntentNewQuestion)
	}
	if tc.QuestionType != QuestionTypeMCQ {
		t.Errorf("question_type = %q; want %q (default)", tc.QuestionType, QuestionTypeMCQ)
	}
	if tc.TenantID != "tenant-x" || tc.AuthorGCID != "gcid-x" {
		t.Errorf("identity pass-through failed: %+v", tc)
	}
}

func TestBuildTaskContextFromState_readsOEQuestionType(t *testing.T) {
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "oe",
		"input_payload": "Explain photosynthesis",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.QuestionType != QuestionTypeOE {
		t.Errorf("question_type = %q; want %q", tc.QuestionType, QuestionTypeOE)
	}
	if tc.Intent != IntentNewQuestion {
		t.Errorf("intent = %q; want new_question", tc.Intent)
	}
	if tc.Prompt != "Explain photosynthesis" {
		t.Errorf("prompt = %q; want %q", tc.Prompt, "Explain photosynthesis")
	}
}

func TestBuildTaskContextFromState_rejectsInvalidQuestionType(t *testing.T) {
	// assert_valid_question_type at the helper level — anything outside
	// {mcq, oe} fails loud (the prior boot-time hardcode silently coerced
	// every value to mcq).
	cases := []string{"essay", "true_false", "MCQ", "OE", "drag-drop", " "}
	for _, q := range cases {
		t.Run("qt="+q, func(t *testing.T) {
			_, err := BuildTaskContextFromState(newFakeState(map[string]any{
				"tenant_id":     "tenant-x",
				"author_gcid":   "gcid-x",
				"question_type": q,
			}))
			if err == nil {
				t.Errorf("BuildTaskContextFromState accepted invalid question_type %q", q)
			}
		})
	}
}

func TestBuildTaskContextFromState_modelAnswerFillSurfacesAuthorMCQOptions(t *testing.T) {
	// _build_session_state writes author_options as a list-of-maps (the
	// Python executor json-encodes/decodes via ReasoningEngine, so by the
	// time we read it back the items are map[string]any).
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "model_answer_fill",
		"question_type": "mcq",
		"author_stem":   "Which Scrum role owns the backlog?",
		"author_options": []any{
			map[string]any{"option_id": "opt-1", "label": "Scrum Master", "is_correct": false},
			map[string]any{"option_id": "opt-2", "label": "Product Owner", "is_correct": true},
		},
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.Intent != IntentModelAnswerFill {
		t.Errorf("intent = %q; want model_answer_fill", tc.Intent)
	}
	if tc.ExistingQuestion == nil {
		t.Fatalf("ExistingQuestion is nil — fill path requires it")
	}
	if tc.ExistingQuestion.Stem != "Which Scrum role owns the backlog?" {
		t.Errorf("Stem = %q; want preserved verbatim", tc.ExistingQuestion.Stem)
	}
	if len(tc.ExistingQuestion.MCQOptions) != 2 {
		t.Errorf("MCQOptions count = %d; want 2", len(tc.ExistingQuestion.MCQOptions))
	}
	if tc.ExistingQuestion.MCQOptions[1].Label != "Product Owner" ||
		!tc.ExistingQuestion.MCQOptions[1].IsCorrect {
		t.Errorf("MCQOptions[1] decoded incorrectly: %+v", tc.ExistingQuestion.MCQOptions[1])
	}
}

func TestBuildTaskContextFromState_modelAnswerFillSurfacesAuthorOERubric(t *testing.T) {
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "model_answer_fill",
		"question_type": "oe",
		"author_stem":   "Explain story points",
		"author_rubric": []any{
			map[string]any{"criterion": "Defines relativity", "weight": 0.5},
			map[string]any{"criterion": "Captures complexity", "weight": 0.5},
		},
		"model_answer": "Story points capture relative effort and complexity...",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.ExistingQuestion == nil {
		t.Fatalf("ExistingQuestion is nil — OE fill path requires it")
	}
	if len(tc.ExistingQuestion.OERubric) != 2 {
		t.Errorf("OERubric count = %d; want 2", len(tc.ExistingQuestion.OERubric))
	}
	var sum float64
	for _, c := range tc.ExistingQuestion.OERubric {
		sum += c.Weight
	}
	if delta := sum - 1.0; delta < -0.01 || delta > 0.01 {
		t.Errorf("OERubric weights sum %v; want 1.0 ±0.01", sum)
	}
	if tc.ExistingQuestion.ModelAnswer == "" {
		t.Errorf("ModelAnswer dropped on round-trip")
	}
}

func TestBuildTaskContextFromState_falsBackToUserGCID(t *testing.T) {
	// manaplugin writes both `user_gcid` (canonical) and the executor
	// writes `author_gcid` (semantically correct for qgen). If only
	// user_gcid is present (e.g., a degraded path where the executor
	// fix did not land yet), we still pull an identity.
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id": "tenant-x",
		"user_gcid": "gcid-fallback",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.AuthorGCID != "gcid-fallback" {
		t.Errorf("AuthorGCID = %q; want user_gcid fallback", tc.AuthorGCID)
	}
}

func TestBuildTaskContextFromState_rejectsNilState(t *testing.T) {
	_, err := BuildTaskContextFromState(nil)
	if err == nil {
		t.Errorf("BuildTaskContextFromState accepted nil state")
	}
}

// -----------------------------------------------------------------------------
// Runtime instruction composition — end-to-end M14.2 contract
// -----------------------------------------------------------------------------

func TestComposeQGenQuestionInstruction_runtimeOEFromState(t *testing.T) {
	// State carrying question_type=oe + intent=new_question MUST resolve
	// to a generation prompt containing the nested oe_payload + rubric
	// invariants — the load-bearing assertion the OE smoke 4/4 depends on.
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "oe",
		"input_payload": "Explain photosynthesis",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	for _, want := range []string{"oe_payload", "rubric", "grader_tier", "40 words"} {
		if !strings.Contains(got, want) {
			t.Errorf("OE-from-state prompt missing %q — runtime branching broken", want)
		}
	}
	// Negative — MCQ-specific phrasing MUST NOT leak into the OE prompt.
	if strings.Contains(got, "1 marked correct") || strings.Contains(got, "distractor") {
		t.Errorf("OE-from-state prompt contains MCQ phrasing — branching broken")
	}
}

func TestComposeQGenQuestionInstruction_runtimeMCQFromState(t *testing.T) {
	// State carrying question_type=mcq (default-equivalent) MUST resolve
	// to a generation prompt containing the MCQ-specific phrasing
	// (options + distractors).
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "mcq",
		"input_payload": "Generate an MCQ on Scrum sprints",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, "options") {
		t.Errorf("MCQ-from-state prompt missing 'options'")
	}
	if !strings.Contains(strings.ToLower(got), "distractor") {
		t.Errorf("MCQ-from-state prompt missing 'distractor'")
	}
	// Negative — OE-specific phrasing MUST NOT leak into MCQ prompt.
	if strings.Contains(got, "oe_payload") || strings.Contains(got, "grader_tier") {
		t.Errorf("MCQ-from-state prompt contains OE phrasing — branching broken")
	}
}

func TestComposeQGenQuestionInstruction_runtimeModelAnswerFillPreservesAuthorStem(t *testing.T) {
	authorStem := "What is the time complexity of binary search?"
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
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
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, authorStem) {
		t.Errorf("fill-MCQ prompt does not surface the author's verbatim stem: %q\n--- prompt:\n%s",
			authorStem, got)
	}
	if !strings.Contains(got, "O(log n)") {
		t.Errorf("fill-MCQ prompt does not surface the author's correct-option label")
	}
}

// -----------------------------------------------------------------------------
// Span attribute contract — shared with 6-agent crew (single crew-wide D6 P4)
// -----------------------------------------------------------------------------

func TestMandatorySpanAttributesShared(t *testing.T) {
	// The 3-agent crew honours the same D6 P4 attribute contract as the
	// 6-agent crew. We don't re-implement MandatorySpanAttributes — we
	// verify the existing list covers the canonical 3-agent shape.
	required := []string{
		"chora.tenant_id",
		"chora.batch_id",
		"chora.crew_kind",
		"chora.qgen.step",
		"gen_ai.request.model",
		"gen_ai.usage.output_tokens",
	}
	got := MandatorySpanAttributes()
	gotSet := make(map[string]struct{}, len(got))
	for _, k := range got {
		gotSet[k] = struct{}{}
	}
	for _, k := range required {
		if _, ok := gotSet[k]; !ok {
			t.Errorf("MandatorySpanAttributes() missing %q (D6 P4 contract drift on qgen_question)", k)
		}
	}
}

// -----------------------------------------------------------------------------
// ADR-197 M-B — operator SAFE-block overrides (read side)
//
// The byte-identical-with-no-overrides golden proof is
// TestComposeQGenQuestion_setModeFalse_goldenByteIdentical (frozen pre-change
// goldens in testdata/). These add focused inertness + substitution coverage.
// -----------------------------------------------------------------------------

func TestComposeQGenQuestion_overridesNil_behaviourNeutral(t *testing.T) {
	// With no SAFE-block override the composed prompt is identical whether
	// Overrides is nil, an empty map, a map with only non-SAFE keys, or blank
	// SAFE values — overrideOr always falls through to the embedded block.
	step := StepGeneration3()
	base := ComposeQGenQuestionInstruction(step, ctxNewMCQ())

	empty := ctxNewMCQ()
	empty.Overrides = map[string]string{}
	if got := ComposeQGenQuestionInstruction(step, empty); got != base {
		t.Errorf("empty overrides changed the prompt:\n%s", firstDiff(base, got))
	}

	// "output"/"context"/"safety" are NOT in the SAFE set — must be ignored so
	// the [EXPECTED OUTPUT] JSON-contract block + safety preamble stay immutable.
	nonSafe := ctxNewMCQ()
	nonSafe.Overrides = map[string]string{"output": "HACKED", "safety": "HACKED", "context": "HACKED"}
	if got := ComposeQGenQuestionInstruction(step, nonSafe); got != base {
		t.Errorf("non-SAFE override keys leaked into the prompt:\n%s", firstDiff(base, got))
	}

	blank := ctxNewMCQ()
	blank.Overrides = map[string]string{"role": "", "task": "", "examples": ""}
	if got := ComposeQGenQuestionInstruction(step, blank); got != base {
		t.Errorf("blank (empty-string) SAFE overrides changed the prompt:\n%s", firstDiff(base, got))
	}
}

func TestComposeQGenQuestion_overridesSubstituteSafeBlocks(t *testing.T) {
	step := StepGeneration3()
	base := ComposeQGenQuestionInstruction(step, ctxNewMCQ())

	ctx := ctxNewMCQ()
	ctx.Overrides = map[string]string{
		"role":     "OVERRIDE_ROLE_SENTINEL",
		"task":     "OVERRIDE_TASK_SENTINEL",
		"examples": "OVERRIDE_EXAMPLES_SENTINEL",
	}
	got := ComposeQGenQuestionInstruction(step, ctx)

	if !strings.Contains(got, "## [ROLE]\nOVERRIDE_ROLE_SENTINEL\n\n") {
		t.Errorf("role override not applied at [ROLE]; got:\n%s", got)
	}
	if !strings.Contains(got, "## [TASK]\nOVERRIDE_TASK_SENTINEL\n\n") {
		t.Errorf("task override not applied at [TASK]; got:\n%s", got)
	}
	if !strings.Contains(got, "## [EXAMPLES]\nOVERRIDE_EXAMPLES_SENTINEL") {
		t.Errorf("examples override not applied at [EXAMPLES]; got:\n%s", got)
	}
	// Substitution, not append — the embedded role block is gone.
	if strings.Contains(got, step.RoleBlock) {
		t.Errorf("embedded role block still present after override (appended, not substituted)")
	}

	// The [EXPECTED OUTPUT] JSON-contract block + the universal "NEVER fabricate"
	// safety preamble are NEVER overridable — the tail from that header must be
	// byte-identical between the base and the overridden prompt.
	const marker = "## [EXPECTED OUTPUT]\n"
	baseTail := base[strings.Index(base, marker):]
	gotTail := got[strings.Index(got, marker):]
	if baseTail != gotTail {
		t.Errorf("[EXPECTED OUTPUT] block changed under overrides — it must be immutable:\n%s",
			firstDiff(baseTail, gotTail))
	}
}

func TestBuildTaskContextFromState_readsOverridesAndVersion(t *testing.T) {
	// prompt_overrides_json tolerated as a JSON STRING and as a native map.
	jsonStr := newFakeState(map[string]any{
		"tenant_id":               "t",
		"question_type":           "mcq",
		"prompt_overrides_json":   `{"role":"R","task":"T"}`,
		"resolved_prompt_version": "v9",
	})
	tc, err := BuildTaskContextFromState(jsonStr)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if tc.Overrides["role"] != "R" || tc.Overrides["task"] != "T" {
		t.Errorf("overrides not read from JSON string: %#v", tc.Overrides)
	}
	if tc.ResolvedPromptVersion != "v9" {
		t.Errorf("resolved_prompt_version=%q want v9", tc.ResolvedPromptVersion)
	}

	native := newFakeState(map[string]any{
		"tenant_id":             "t",
		"question_type":         "mcq",
		"prompt_overrides_json": map[string]any{"examples": "E"},
	})
	tc2, err := BuildTaskContextFromState(native)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if tc2.Overrides["examples"] != "E" {
		t.Errorf("overrides not read from native map: %#v", tc2.Overrides)
	}

	// Malformed → nil (fail-soft, never an error, never a panic).
	bad := newFakeState(map[string]any{
		"tenant_id":             "t",
		"question_type":         "mcq",
		"prompt_overrides_json": `{not json`,
	})
	tc3, err := BuildTaskContextFromState(bad)
	if err != nil {
		t.Fatalf("malformed overrides must not error: %v", err)
	}
	if tc3.Overrides != nil {
		t.Errorf("malformed overrides must degrade to nil; got %#v", tc3.Overrides)
	}
}
