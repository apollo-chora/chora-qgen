// Wire-form tolerance + degradation tests for the session-state readers.
//
// The Python executor and the ReasoningEngine JSON round-trip do not agree on a
// single representation for the compound state keys: a list may arrive as a
// json.dumps STRING or as a native []any of maps, an override map may arrive as
// a JSON object STRING or as a native map. Each reader therefore declares a
// posture the composer depends on, and the two postures are deliberately
// DIFFERENT:
//
//   - type_plan_json is LOAD-BEARING in set mode, so a present-but-broken plan
//     is a hard error (a set turn must never silently degrade to one candidate).
//   - avoid_concepts / prompt_overrides / author_options / author_rubric are
//     advisory, so a broken value degrades to "no hint" and the turn proceeds
//     with the embedded defaults.
//
// These tests pin both postures against every wire form the readers claim to
// tolerate, since a reader that quietly returned nil for the load-bearing key
// would turn a set-mode contract break into a wrong-shaped prompt.
package agent

import (
	"strings"
	"testing"
)

// setModeState builds a valid set-mode state (the only mode in which
// type_plan_json / avoid_concepts_json are read at all) with the given keys
// layered on top.
func setModeState(extra map[string]any) *mapState {
	m := map[string]any{
		"tenant_id":      "t",
		"author_gcid":    "g",
		"set_mode":       true,
		"type_plan_json": `[{"question_type":"mcq","count":2,"max_images":1}]`,
	}
	for k, v := range extra {
		m[k] = v
	}
	return &mapState{m: m}
}

// -----------------------------------------------------------------------------
// avoid_concepts_json : the regenerate-rejected dedup hint (advisory)
// -----------------------------------------------------------------------------

func TestAvoidConcepts_nativeStringSlice(t *testing.T) {
	// A native []string from a Go caller passes through unchanged.
	tc, err := BuildTaskContextFromState(setModeState(map[string]any{
		"avoid_concepts_json": []string{"evaporation", "condensation"},
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if len(tc.AvoidConcepts) != 2 || tc.AvoidConcepts[0] != "evaporation" || tc.AvoidConcepts[1] != "condensation" {
		t.Errorf("native []string not carried: %#v", tc.AvoidConcepts)
	}
}

func TestAvoidConcepts_nativeAnySliceKeepsOnlyNonBlankStrings(t *testing.T) {
	// The ReasoningEngine round-trip hands back []any. Non-strings and
	// whitespace-only entries carry no concept to avoid, so they are dropped
	// rather than injected into the [DIVERSITY] block as empty avoidances.
	tc, err := BuildTaskContextFromState(setModeState(map[string]any{
		"avoid_concepts_json": []any{"photosynthesis", 42, "   ", nil, "respiration"},
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if len(tc.AvoidConcepts) != 2 {
		t.Fatalf("want 2 surviving concepts, got %#v", tc.AvoidConcepts)
	}
	if tc.AvoidConcepts[0] != "photosynthesis" || tc.AvoidConcepts[1] != "respiration" {
		t.Errorf("wrong survivors (order must be preserved): %#v", tc.AvoidConcepts)
	}
}

func TestAvoidConcepts_bareStringIsNotAConcept(t *testing.T) {
	// A plain string is not a JSON array. Treating it as a one-element list
	// would smuggle a json.dumps artefact into the prompt, so it degrades to
	// no avoidance.
	tc, err := BuildTaskContextFromState(setModeState(map[string]any{
		"avoid_concepts_json": "evaporation",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.AvoidConcepts != nil {
		t.Errorf("bare string must not become a concept list; got %#v", tc.AvoidConcepts)
	}
}

func TestAvoidConcepts_malformedJSONArrayDegradesToNone(t *testing.T) {
	// Looks like an array, does not parse. Advisory key, so the set turn still
	// runs with no extra avoidance instead of failing.
	tc, err := BuildTaskContextFromState(setModeState(map[string]any{
		"avoid_concepts_json": `["evaporation",`,
	}))
	if err != nil {
		t.Fatalf("a malformed advisory hint must not fail the turn: %v", err)
	}
	if tc.AvoidConcepts != nil {
		t.Errorf("malformed array must yield no concepts; got %#v", tc.AvoidConcepts)
	}
}

func TestAvoidConcepts_unrecognisedShapeDegradesToNone(t *testing.T) {
	// Neither a slice nor a string: nothing to read, nothing to avoid.
	tc, err := BuildTaskContextFromState(setModeState(map[string]any{
		"avoid_concepts_json": 42,
	}))
	if err != nil {
		t.Fatalf("an unrecognised advisory shape must not fail the turn: %v", err)
	}
	if tc.AvoidConcepts != nil {
		t.Errorf("unrecognised shape must yield no concepts; got %#v", tc.AvoidConcepts)
	}
}

// -----------------------------------------------------------------------------
// type_plan_json : load-bearing in set mode, so it fails loud
// -----------------------------------------------------------------------------

func TestTypePlan_emptyStringFailsLoudInSetMode(t *testing.T) {
	// An empty type_plan_json reads as "no plan". In set mode that is a
	// contract break: falling back to a single candidate would silently ship
	// the author one question instead of the set they paid mana for.
	_, err := BuildTaskContextFromState(setModeState(map[string]any{
		"type_plan_json": "   ",
	}))
	if err == nil {
		t.Fatal("set mode with a blank type_plan_json must fail loud")
	}
	if !strings.Contains(err.Error(), "non-empty type_plan") {
		t.Errorf("error must name the empty plan; got %v", err)
	}
}

func TestTypePlan_unmarshalableNativeValueFailsLoud(t *testing.T) {
	// A native value the JSON round-trip cannot represent (a channel) must
	// surface as a re-marshal error, not be skipped as "absent".
	_, err := BuildTaskContextFromState(setModeState(map[string]any{
		"type_plan_json": make(chan int),
	}))
	if err == nil {
		t.Fatal("an unmarshalable type_plan must fail loud")
	}
	if !strings.Contains(err.Error(), "re-marshal") {
		t.Errorf("error must identify the re-marshal failure; got %v", err)
	}
}

// -----------------------------------------------------------------------------
// Boolean state keys : native bool, string form, anything else
// -----------------------------------------------------------------------------

func TestStateBool_stringTrueOptsTheImageIn(t *testing.T) {
	// A caller that stamps string state still opts the author in: "true" is
	// the only accepted string form.
	tc, err := BuildTaskContextFromState(&mapState{m: map[string]any{
		"intent":           "new_question",
		"question_type":    "mcq",
		"image_for_stem":   "true",
		"image_for_answer": "false",
	}})
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if !tc.ImageForStem {
		t.Error("image_for_stem=\"true\" must opt the stem image in")
	}
	if tc.ImageForAnswer {
		t.Error("image_for_answer=\"false\" must stay opted out")
	}
}

func TestStateBool_otherStringsDoNotOptIn(t *testing.T) {
	// "yes" / "1" are not the accepted form. Guessing at them would charge the
	// author for an image they never asked for.
	tc, err := BuildTaskContextFromState(&mapState{m: map[string]any{
		"intent":           "new_question",
		"question_type":    "mcq",
		"image_for_stem":   "yes",
		"image_for_answer": "1",
	}})
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.ImageForStem || tc.ImageForAnswer {
		t.Errorf("only \"true\" opts in; got stem=%v answer=%v", tc.ImageForStem, tc.ImageForAnswer)
	}
}

func TestStateBool_nonBooleanShapeDoesNotOptIn(t *testing.T) {
	// A numeric 1 is neither a bool nor the string "true": fail-soft to no
	// image guidance rather than inventing an opt-in.
	tc, err := BuildTaskContextFromState(&mapState{m: map[string]any{
		"intent":         "new_question",
		"question_type":  "mcq",
		"image_for_stem": 1,
	}})
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.ImageForStem {
		t.Error("a numeric image_for_stem must not opt in")
	}
}

// -----------------------------------------------------------------------------
// prompt_overrides_json : ADR-197 M-B SAFE-block overrides (advisory)
// -----------------------------------------------------------------------------

func overrideCtxFromState(t *testing.T, v any) TaskContextQuestion {
	t.Helper()
	tc, err := BuildTaskContextFromState(&mapState{m: map[string]any{
		"intent":                "new_question",
		"question_type":         "mcq",
		"prompt_overrides_json": v,
	}})
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	return tc
}

func TestPromptOverrides_blankStringYieldsEmbeddedBlocks(t *testing.T) {
	tc := overrideCtxFromState(t, "   ")
	if tc.Overrides != nil {
		t.Errorf("a blank override payload must leave the embedded blocks in place; got %#v", tc.Overrides)
	}
	// The composed prompt must be the embedded role, not an empty [ROLE].
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, StepGeneration3().RoleBlock) {
		t.Error("embedded RoleBlock must survive a blank override payload")
	}
}

func TestPromptOverrides_nativeStringMapIsApplied(t *testing.T) {
	// A Go caller may stamp map[string]string directly rather than a JSON blob.
	tc := overrideCtxFromState(t, map[string]string{"role": "OPERATOR ROLE BLOCK"})
	if tc.Overrides["role"] != "OPERATOR ROLE BLOCK" {
		t.Fatalf("native map[string]string not carried: %#v", tc.Overrides)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, "OPERATOR ROLE BLOCK") {
		t.Error("the operator role override must reach the composed [ROLE] block")
	}
	if strings.Contains(got, StepGeneration3().RoleBlock) {
		t.Error("the embedded RoleBlock must be REPLACED, not appended to")
	}
}

func TestPromptOverrides_emptyNativeMapYieldsEmbeddedBlocks(t *testing.T) {
	tc := overrideCtxFromState(t, map[string]string{})
	if tc.Overrides != nil {
		t.Errorf("an empty override map must read as no overrides; got %#v", tc.Overrides)
	}
}

func TestPromptOverrides_emptyJSONObjectYieldsEmbeddedBlocks(t *testing.T) {
	// json.dumps({}) is the wire form of "operator set no overrides".
	tc := overrideCtxFromState(t, `{}`)
	if tc.Overrides != nil {
		t.Errorf("an empty JSON object must read as no overrides; got %#v", tc.Overrides)
	}
}

func TestPromptOverrides_unmarshalableValueDegradesToEmbedded(t *testing.T) {
	// Overrides are advisory: an unrepresentable value must fall back to the
	// embedded blocks rather than fail the author's turn.
	tc := overrideCtxFromState(t, make(chan int))
	if tc.Overrides != nil {
		t.Errorf("an unmarshalable override payload must degrade to nil; got %#v", tc.Overrides)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, StepGeneration3().RoleBlock) {
		t.Error("embedded RoleBlock must survive an unmarshalable override payload")
	}
}

// -----------------------------------------------------------------------------
// author_options / author_rubric : the model_answer_fill inputs (best effort)
// -----------------------------------------------------------------------------

func fillState(qt string, extra map[string]any) *mapState {
	m := map[string]any{
		"tenant_id":     "t",
		"author_gcid":   "g",
		"intent":        "model_answer_fill",
		"question_type": qt,
		"author_stem":   "Which gas do plants absorb?",
	}
	for k, v := range extra {
		m[k] = v
	}
	return &mapState{m: m}
}

func TestFillMCQ_absentAuthorOptionsLeavesNoOptions(t *testing.T) {
	// A fill turn that never stamped author_options has nothing to preserve.
	// The ExistingQuestion still exists (the stem is the fill anchor), but the
	// option list must be empty rather than fabricated.
	tc, err := BuildTaskContextFromState(fillState("mcq", nil))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.ExistingQuestion == nil {
		t.Fatal("a fill turn must carry an ExistingQuestion")
	}
	if len(tc.ExistingQuestion.MCQOptions) != 0 {
		t.Errorf("absent author_options must yield no options; got %#v", tc.ExistingQuestion.MCQOptions)
	}
}

func TestFillMCQ_nonListAuthorOptionsYieldsNoOptions(t *testing.T) {
	tc, err := BuildTaskContextFromState(fillState("mcq", map[string]any{
		"author_options": "a,b,c",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if len(tc.ExistingQuestion.MCQOptions) != 0 {
		t.Errorf("a non-list author_options must yield no options; got %#v", tc.ExistingQuestion.MCQOptions)
	}
}

func TestFillMCQ_skipsMalformedOptionItems(t *testing.T) {
	// Best-effort decode: the well-formed options are preserved verbatim (the
	// fill contract is that author options are NEVER rewritten) and the junk
	// entry is dropped instead of failing the whole turn.
	tc, err := BuildTaskContextFromState(fillState("mcq", map[string]any{
		"author_options": []any{
			map[string]any{"option_id": "a", "label": "Oxygen", "is_correct": true},
			"not-an-option",
			map[string]any{"option_id": "b", "label": "Carbon dioxide", "is_correct": false},
		},
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	opts := tc.ExistingQuestion.MCQOptions
	if len(opts) != 2 {
		t.Fatalf("want the 2 well-formed options, got %#v", opts)
	}
	if opts[0].OptionID != "a" || opts[0].Label != "Oxygen" || !opts[0].IsCorrect {
		t.Errorf("first option mangled: %#v", opts[0])
	}
	if opts[1].OptionID != "b" || opts[1].IsCorrect {
		t.Errorf("second option mangled: %#v", opts[1])
	}
}

func TestFillOE_absentAuthorRubricLeavesNoRubric(t *testing.T) {
	tc, err := BuildTaskContextFromState(fillState("oe", nil))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.ExistingQuestion == nil {
		t.Fatal("a fill turn must carry an ExistingQuestion")
	}
	if len(tc.ExistingQuestion.OERubric) != 0 {
		t.Errorf("absent author_rubric must yield no criteria; got %#v", tc.ExistingQuestion.OERubric)
	}
}

func TestFillOE_nonListAuthorRubricYieldsNoRubric(t *testing.T) {
	tc, err := BuildTaskContextFromState(fillState("oe", map[string]any{
		"author_rubric": 42,
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if len(tc.ExistingQuestion.OERubric) != 0 {
		t.Errorf("a non-list author_rubric must yield no criteria; got %#v", tc.ExistingQuestion.OERubric)
	}
}

func TestFillOE_skipsMalformedRubricItems(t *testing.T) {
	tc, err := BuildTaskContextFromState(fillState("oe", map[string]any{
		"author_rubric": []any{
			map[string]any{"criterion": "Accuracy", "weight": 0.6},
			nil,
			map[string]any{"criterion": "Clarity", "weight": 0.4},
		},
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	rub := tc.ExistingQuestion.OERubric
	if len(rub) != 2 {
		t.Fatalf("want the 2 well-formed criteria, got %#v", rub)
	}
	if rub[0].Criterion != "Accuracy" || rub[0].Weight != 0.6 {
		t.Errorf("first criterion mangled: %#v", rub[0])
	}
	if rub[1].Criterion != "Clarity" || rub[1].Weight != 0.4 {
		t.Errorf("second criterion mangled: %#v", rub[1])
	}
}

// -----------------------------------------------------------------------------
// asFloat : the numeric coercion every weight / counter read goes through
// -----------------------------------------------------------------------------

func TestAsFloat_numericWideningAndNonNumericZero(t *testing.T) {
	// The executor's JSON path hands float64; native Go callers hand int /
	// int64 / float32. Anything non-numeric is 0, which is what makes a
	// string-typed weight read as "unweighted" rather than panicking.
	cases := []struct {
		name string
		in   any
		want float64
	}{
		{"float64 from the JSON round-trip", float64(0.25), 0.25},
		{"float32 from a native caller", float32(0.5), 0.5},
		{"int from a native caller", int(3), 3},
		{"int64 from a native caller", int64(7), 7},
		{"string is not coerced", "0.9", 0},
		{"nil is not coerced", nil, 0},
		{"bool is not coerced", true, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := asFloat(tc.in); got != tc.want {
				t.Errorf("asFloat(%#v) = %v; want %v", tc.in, got, tc.want)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// image_regen edge shapes
// -----------------------------------------------------------------------------

func TestImageRegen_missingPlacementDefaultsToStem(t *testing.T) {
	// A regen turn that never stamped a placement must target the stem rather
	// than emit an empty placement into the image_spec the orchestrator
	// renders.
	tc, err := BuildTaskContextFromState(imageRegenState("", nil))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	// Assert on the [IMAGE] directive, NOT on `"placement": "stem"`: the
	// [EXPECTED OUTPUT] schema line carries "stem"|"answer" verbatim and would
	// match that substring whatever the resolved placement is.
	if !strings.Contains(got, "for the stem of this question") {
		t.Errorf("blank placement must default to the stem image\n%s", got)
	}
	if strings.Contains(got, "for the answer of this question") {
		t.Errorf("a blank placement must not be resolved to the answer\n%s", got)
	}
	if strings.Contains(got, "for the  of this question") {
		t.Errorf("an empty placement must never reach the prompt\n%s", got)
	}
	// The [TASK] tells the model what it is re-authoring and must agree.
	if !strings.Contains(got, "Re-author ONLY the stem illustration") {
		t.Errorf("the [TASK] must default to the stem illustration too\n%s", got)
	}
}

func TestImageRegen_explicitPlacementIsHonoured(t *testing.T) {
	// Positive control for the default above: a stamped placement is used
	// as-is, so "stem" is a fallback and not a hardcode.
	tc, err := BuildTaskContextFromState(imageRegenState("answer", nil))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, "for the answer of this question") {
		t.Errorf("an explicit answer placement must be honoured\n%s", got)
	}
	if strings.Contains(got, "for the stem of this question") {
		t.Errorf("an answer regen must not also target the stem\n%s", got)
	}
}

func TestImageRegen_withoutRegenContextEmitsNoImageBlock(t *testing.T) {
	// Defensive shape: image_regen intent with no regen detail attached. The
	// composer must drop the [IMAGE] block rather than render a half-formed
	// one, and must not panic dereferencing the missing context.
	got := ComposeQGenQuestionInstruction(StepGeneration3(), TaskContextQuestion{
		TenantID:     "t",
		AuthorGCID:   "g",
		Intent:       IntentImageRegen,
		QuestionType: QuestionTypeMCQ,
	})
	if strings.Contains(got, "## [IMAGE]") {
		t.Errorf("no regen context must mean no [IMAGE] block\n%s", got)
	}
	// The regen TASK + OUTPUT still stand, so the turn stays a single-spec
	// re-author and never degrades into a question re-draft.
	if !strings.Contains(got, "image regenerate") {
		t.Errorf("the image_regen task must survive a missing regen context\n%s", got)
	}
	if !strings.Contains(got, "image_spec") {
		t.Errorf("the single-image_spec output contract must survive\n%s", got)
	}
}

// -----------------------------------------------------------------------------
// resolveTaskAndOutput : the unknown-combination refusal
// -----------------------------------------------------------------------------

func TestResolveTaskAndOutput_unknownIntentRefusesGeneration(t *testing.T) {
	// BuildTaskContextFromState validates question_type but NOT intent, so an
	// unrecognised intent reaches the composer. It must refuse rather than
	// fall back to the new-MCQ template and hand the author a question they
	// did not ask for.
	got := ComposeQGenQuestionInstruction(StepGeneration3(), TaskContextQuestion{
		TenantID:     "t",
		AuthorGCID:   "g",
		Intent:       Intent("rewrite_everything"),
		QuestionType: QuestionTypeMCQ,
	})
	if !strings.Contains(got, "refuse generation") {
		t.Errorf("an unknown intent must refuse generation\n%s", got)
	}
	if !strings.Contains(got, "invalid_intent_question_type_combination") {
		t.Errorf("the refusal must carry the explicit error contract\n%s", got)
	}
	if strings.Contains(got, taskNewMCQ()) {
		t.Errorf("an unknown intent must NOT fall back to the new-MCQ template\n%s", got)
	}
}

func TestResolveTaskAndOutput_unknownQuestionTypeRefusesGeneration(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), TaskContextQuestion{
		TenantID:     "t",
		AuthorGCID:   "g",
		Intent:       IntentNewQuestion,
		QuestionType: QuestionType("true_false"),
	})
	if !strings.Contains(got, "invalid_intent_question_type_combination") {
		t.Errorf("an unsupported question_type must refuse generation\n%s", got)
	}
}
