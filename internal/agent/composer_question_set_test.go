// Set-mode (mixed-type batch) composer tests for the qgen_question generation
// agent — CHO-1819 P1c.
//
// Set mode flips the generation step from emitting ONE candidate to a SET of
// mixed-type candidates wrapped in {candidates, generation_summary}. The hard
// contract is that the LIVE single-candidate path (SetMode=false) is byte-for-
// byte UNCHANGED — proven by TestComposeQGenQuestion_setModeFalse_goldenByteIdentical
// against goldens captured from the pre-P1c code in testdata/.
//
// Coverage:
//   - Golden byte-equality of the SetMode=false generation prompt (legacy path
//     untouched) + a defence-in-depth "new fields are inert" check.
//   - readStateInt table tests (the shared int state reader).
//   - Each set block builder (setPlanBlock / diversityBlock / imageBudgetBlock /
//     setTask / setOutputSchema) on the canonical 8-MCQ-3-img + 2-OE-1-img plan.
//   - Full ComposeQGenQuestionInstruction in set mode: block presence + order,
//     the candidates+generation_summary wrapper, per-item question_type, the
//     canonical generation_summary proto field names, AvoidConcepts rendering,
//     and the legacy [IMAGE] opt-in block being replaced by [IMAGE BUDGET].
//   - BuildTaskContextFromState set-mode parsing (JSON-string + native plan
//     forms) + fail-loud on empty / invalid / malformed plans, and the single
//     path ignoring a stray type_plan_json entirely.
package agent

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ctxSet8mcq2oe is the canonical mixed plan used across the block-builder tests:
// 8 MCQ (≤3 images) + 2 OE (≤1 image) = 10 questions.
func ctxSet8mcq2oe() TaskContextQuestion {
	return TaskContextQuestion{
		TenantID:   "tenant-x",
		AuthorGCID: "gcid-x",
		Intent:     IntentNewQuestion,
		Prompt:     "Generate a mixed set on the water cycle",
		SetMode:    true,
		TypePlan: []GenerationTypeQuota{
			{QuestionType: "mcq", Count: 8, MaxImages: 3},
			{QuestionType: "oe", Count: 2, MaxImages: 1},
		},
	}
}

// -----------------------------------------------------------------------------
// (1) Golden byte-equality — the SetMode=false legacy path is UNCHANGED.
// -----------------------------------------------------------------------------

// updateGolden regenerates the testdata/golden_gen_*.txt files when an
// INTENTIONAL prompt change lands (run: go test ./internal/agent/ -update).
// CHO-1822: the forced-image golden (new_mcq_img_both) was regenerated when
// forced images became INTEGRAL (Item 1) — the no-image goldens are unaffected.
var updateGolden = flag.Bool("update", false, "regenerate golden_gen_*.txt files")

func TestComposeQGenQuestion_setModeFalse_goldenByteIdentical(t *testing.T) {
	// Goldens were frozen from the PRE-P1c code (single-candidate generation
	// prompts). Any drift here means the legacy path is no longer byte-identical
	// — exactly the regression P1c must not introduce.
	cases := map[string]TaskContextQuestion{
		"new_mcq":          ctxNewMCQ(),
		"new_oe":           ctxNewOE(),
		"fill_mcq":         ctxFillMCQ(),
		"fill_oe":          ctxFillOE(),
		"new_mcq_img_both": ctxNewMCQWithImage(true, true),
	}
	for name, ctx := range cases {
		t.Run(name, func(t *testing.T) {
			got := ComposeQGenQuestionInstruction(StepGeneration3(), ctx)
			path := filepath.Join("testdata", "golden_gen_"+name+".txt")
			if *updateGolden {
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatalf("write golden %s: %v", path, err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read golden %s: %v", path, err)
			}
			if got != string(want) {
				t.Errorf("SetMode=false %s drifted from the captured legacy golden — the "+
					"single-candidate path is NOT byte-identical.\n%s", name, firstDiff(string(want), got))
			}
		})
	}
}

func TestComposeQGenQuestion_setModeFalse_ignoresSetFields(t *testing.T) {
	// Defence-in-depth on top of the golden: with SetMode=false, populating
	// TypePlan + AvoidConcepts MUST change nothing (they belong to set mode only).
	base := ctxNewMCQ()
	withSetFields := base
	withSetFields.TypePlan = []GenerationTypeQuota{{QuestionType: "mcq", Count: 5, MaxImages: 2}}
	withSetFields.AvoidConcepts = []string{"already covered"}
	a := ComposeQGenQuestionInstruction(StepGeneration3(), base)
	b := ComposeQGenQuestionInstruction(StepGeneration3(), withSetFields)
	if a != b {
		t.Errorf("SetMode=false output changed when TypePlan/AvoidConcepts were populated — "+
			"the new fields leaked into the legacy path.\n%s", firstDiff(a, b))
	}
}

// -----------------------------------------------------------------------------
// (2) readStateInt table tests — the shared int state reader (critic.go).
// -----------------------------------------------------------------------------

func TestReadStateInt_table(t *testing.T) {
	// readStateInt is the shared int state reader (defined in critic.go,
	// delegates to asFloat). Set-mode decodes type_plan via typed json.Unmarshal
	// rather than this helper, but the helper was previously untested — pin its
	// actual (asFloat-backed) behaviour, including string→0 (asFloat does not
	// parse strings; critic.go is the qgen_critic agent and is out of P1c scope).
	cases := []struct {
		name    string
		val     any
		present bool
		want    int
	}{
		{"native int", 7, true, 7},
		{"float64 from JSON round-trip", float64(8), true, 8},
		{"int64", int64(42), true, 42},
		{"missing key defaults 0", nil, false, 0},
		{"string is not coerced (asFloat-backed) → 0", "9", true, 0},
		{"bool → 0", true, true, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := map[string]any{}
			if c.present {
				m["k"] = c.val
			}
			if got := readStateInt(newFakeState(m), "k"); got != c.want {
				t.Errorf("readStateInt(%#v) = %d; want %d", c.val, got, c.want)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// (3) Individual set block builders — exact substrings on the canonical plan.
// -----------------------------------------------------------------------------

func TestSetPlanBlock_countsAndTypes(t *testing.T) {
	got := setPlanBlock(ctxSet8mcq2oe())
	if !strings.HasPrefix(got, "## [SET PLAN]\n") {
		t.Errorf("setPlanBlock missing header; got:\n%s", got)
	}
	for _, want := range []string{
		"Generate a SET of 10 questions",
		"8 of type mcq",
		"2 of type oe",
		"one `candidates` array",
		"each element declares its own `question_type`",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("[SET PLAN] missing %q\n%s", want, got)
		}
	}
}

func TestDiversityBlock_sentenceAndAvoidConcepts(t *testing.T) {
	// Without avoid-concepts: the dedup sentence, and NO avoidance line.
	got := diversityBlock(ctxSet8mcq2oe())
	if !strings.HasPrefix(got, "## [DIVERSITY]\n") {
		t.Errorf("diversityBlock missing header")
	}
	for _, want := range []string{
		"Every question MUST assess a DISTINCT concept",
		"No two may test the same fact/definition/skill/worked-example",
		"This single response is the only deduplication pass",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("[DIVERSITY] missing %q", want)
		}
	}
	if strings.Contains(got, "Do NOT repeat") {
		t.Errorf("[DIVERSITY] leaked an avoidance line when AvoidConcepts is empty")
	}

	// With avoid-concepts: avoidance line appended, concepts joined.
	withAvoid := ctxSet8mcq2oe()
	withAvoid.AvoidConcepts = []string{"evaporation basics", "condensation"}
	got2 := diversityBlock(withAvoid)
	for _, want := range []string{
		"Do NOT repeat any of these already-covered concepts:",
		"evaporation basics",
		"condensation",
	} {
		if !strings.Contains(got2, want) {
			t.Errorf("[DIVERSITY] avoid-concepts missing %q\n%s", want, got2)
		}
	}
}

func TestImageBudgetBlock_perTypeCaps(t *testing.T) {
	got := imageBudgetBlock(ctxSet8mcq2oe())
	if !strings.HasPrefix(got, "## [IMAGE BUDGET]\n") {
		t.Errorf("imageBudgetBlock missing header")
	}
	for _, want := range []string{
		"Across the 8 mcq questions, AT MOST 3 may include an image",
		"Across the 2 oe questions, AT MOST 1 may include an image",
		"`image_specs`",
		"Do NOT exceed the caps",
		// CHO-1825 2a: the budget must be USED, not discouraged. The agent
		// should add images up to the cap where they aid comprehension.
		"USE this budget",
		"Aim to use the full budget",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("[IMAGE BUDGET] missing %q\n%s", want, got)
		}
	}
	// CHO-1825 2a: the AI-decide anti-bias that suppressed batch images (1/7
	// live runs emitted) MUST be gone — these phrases drove the model to emit
	// no images at all.
	for _, banned := range []string{
		"most questions should carry no image",
		"GENUINELY aids comprehension",
	} {
		if strings.Contains(got, banned) {
			t.Errorf("[IMAGE BUDGET] must NOT contain image-suppressing phrase %q\n%s", banned, got)
		}
	}
}

// ctxSetForced is a deterministic per-type image plan (CHO-1825 2b): 2 MCQ that
// MUST each carry a stem image, 1 OE that MUST carry an answer image, ZERO
// discretionary budget. The author forced the images via the per-type toggles.
func ctxSetForced() TaskContextQuestion {
	c := ctxSet8mcq2oe()
	c.TypePlan = []GenerationTypeQuota{
		{QuestionType: "mcq", Count: 2, MaxImages: 0, ImageForStem: true},
		{QuestionType: "oe", Count: 1, MaxImages: 0, ImageForAnswer: true},
	}
	return c
}

func TestImageBudgetBlock_forcedImagesAreDeterministicMustEmit(t *testing.T) {
	got := imageBudgetBlock(ctxSetForced())
	if !strings.HasPrefix(got, "## [IMAGE BUDGET]\n") {
		t.Errorf("imageBudgetBlock missing header")
	}
	for _, want := range []string{
		// EVERY question of a forced type must carry the image — not discretionary.
		"EVERY one of the 2 mcq question(s) MUST carry",
		"placement=\"stem\"",
		"EVERY one of the 1 oe question(s) MUST carry",
		"placement=\"answer\"",
		"`image_specs`",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("[IMAGE BUDGET] forced path missing %q\n%s", want, got)
		}
	}
	// A forced type with MaxImages=0 must NOT be described as an AT-MOST budget
	// (that would let the model emit zero — defeating the deterministic toggle).
	if strings.Contains(got, "AT MOST 0 may include an image") {
		t.Errorf("[IMAGE BUDGET] rendered a forced type as a 0-image budget\n%s", got)
	}
}

func TestImageBudgetBlock_mixedForcedAndDiscretionary(t *testing.T) {
	c := ctxSet8mcq2oe()
	c.TypePlan = []GenerationTypeQuota{
		{QuestionType: "mcq", Count: 4, MaxImages: 2},                      // discretionary
		{QuestionType: "oe", Count: 1, MaxImages: 0, ImageForAnswer: true}, // forced
	}
	got := imageBudgetBlock(c)
	// Discretionary type keeps the AT-MOST budget phrasing + the USE-it nudge.
	if !strings.Contains(got, "Across the 4 mcq questions, AT MOST 2 may include an image") {
		t.Errorf("[IMAGE BUDGET] mixed: discretionary mcq budget missing\n%s", got)
	}
	if !strings.Contains(got, "USE") {
		t.Errorf("[IMAGE BUDGET] mixed: discretionary USE-budget nudge missing\n%s", got)
	}
	// Forced type carries the must-emit instruction.
	if !strings.Contains(got, "EVERY one of the 1 oe question(s) MUST carry") {
		t.Errorf("[IMAGE BUDGET] mixed: forced oe must-emit missing\n%s", got)
	}
}

func TestSetOutputSchema_allowsForcedImageSpecs(t *testing.T) {
	got := setOutputSchema(ctxSetForced())
	for _, want := range []string{
		"\"image_specs\"",
		// Up to two entries (one per placement) so a both-parts-forced candidate fits.
		"up to 2 entries",
		// The schema must point the model at the [IMAGE BUDGET] block's required images.
		"[IMAGE BUDGET]",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("setOutputSchema forced-image allowance missing %q\n%s", want, got)
		}
	}
}

func TestBuildTaskContextFromState_setModeDecodesPerTypeImageFlags(t *testing.T) {
	// CHO-1825 2b — the per-type image_for_stem / image_for_answer toggles ride
	// through type_plan_json and decode into the TypePlan bools the prompt reads.
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":   "tenant-x",
		"author_gcid": "gcid-x",
		"set_mode":    true,
		"type_plan_json": `[{"question_type":"mcq","count":2,"max_images":0,"image_for_stem":true},` +
			`{"question_type":"oe","count":1,"max_images":0,"image_for_answer":true}]`,
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if len(tc.TypePlan) != 2 {
		t.Fatalf("TypePlan len = %d; want 2", len(tc.TypePlan))
	}
	if !tc.TypePlan[0].ImageForStem || tc.TypePlan[0].ImageForAnswer {
		t.Errorf("TypePlan[0] flags = stem:%v answer:%v; want stem:true answer:false",
			tc.TypePlan[0].ImageForStem, tc.TypePlan[0].ImageForAnswer)
	}
	if tc.TypePlan[1].ImageForStem || !tc.TypePlan[1].ImageForAnswer {
		t.Errorf("TypePlan[1] flags = stem:%v answer:%v; want stem:false answer:true",
			tc.TypePlan[1].ImageForStem, tc.TypePlan[1].ImageForAnswer)
	}
}

func TestSetTask_reusesPerTypeContracts(t *testing.T) {
	got := setTask(ctxSet8mcq2oe())
	low := strings.ToLower(got)
	// MCQ contract reused verbatim from taskNewMCQ (distractors).
	if !strings.Contains(low, "distractor") {
		t.Errorf("setTask missing the MCQ distractor contract (taskNewMCQ reuse)\n%s", got)
	}
	// OE contract reused verbatim from taskNewOE (model_answer + rubric).
	if !strings.Contains(low, "model_answer") {
		t.Errorf("setTask missing the OE model_answer contract (taskNewOE reuse)")
	}
	if !strings.Contains(got, "[DIVERSITY]") || !strings.Contains(got, "[IMAGE BUDGET]") {
		t.Errorf("setTask does not point at the [DIVERSITY]/[IMAGE BUDGET] blocks")
	}
}

func TestSetOutputSchema_wrapperAndProtoFieldNames(t *testing.T) {
	got := setOutputSchema(ctxSet8mcq2oe())
	for _, want := range []string{
		"\"candidates\"",
		"\"generation_summary\"",
		"\"question_type\": \"mcq\"",
		"\"question_type\": \"oe\"",
		"\"options\"",
		"\"oe_payload\"",
		"grader_tier",
		// Canonical chora.creation.v1.GenerationSummary proto field names.
		"requested_total",
		"generated_total",
		"generated_per_type",
		"shortfall_reason",
		"generated_total MUST EQUAL the number of",
		// optional per-candidate images + strict shortfall.
		"\"image_specs\"",
		"Never pad, duplicate, or invent",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("setOutputSchema missing %q\n%s", want, got)
		}
	}
}

// -----------------------------------------------------------------------------
// (4) Full set-mode composition — block order + wrapper + isolation.
// -----------------------------------------------------------------------------

func TestComposeQGenQuestion_setMode_blocksInOrder(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxSet8mcq2oe())
	order := []string{
		"## [TASK]",
		"## [SET PLAN]",
		"## [DIVERSITY]",
		"## [IMAGE BUDGET]",
		"## [EXPECTED OUTPUT]",
	}
	last := -1
	for _, blk := range order {
		i := strings.Index(got, blk)
		if i < 0 {
			t.Fatalf("set-mode prompt missing block %q\n%s", blk, got)
		}
		if i <= last {
			t.Errorf("set-mode block %q out of order (at %d, after %d)", blk, i, last)
		}
		last = i
	}
}

func TestComposeQGenQuestion_setMode_omitsLegacyImageOptInBlock(t *testing.T) {
	// In set mode the per-type [IMAGE BUDGET] replaces the single-path
	// author-opt-in [IMAGE] block — even when the legacy flags are set.
	c := ctxSet8mcq2oe()
	c.ImageForStem = true
	c.ImageForAnswer = true
	got := ComposeQGenQuestionInstruction(StepGeneration3(), c)
	if strings.Contains(got, "## [IMAGE]\n") {
		t.Errorf("set mode emitted the legacy [IMAGE] author-opt-in block")
	}
	if !strings.Contains(got, "## [IMAGE BUDGET]") {
		t.Errorf("set mode missing the [IMAGE BUDGET] block")
	}
	// The single-path output fragment must not leak in either.
	if strings.Contains(got, "ADDITIONALLY include an `image_specs` array on the candidate object with EXACTLY") {
		t.Errorf("set mode leaked the single-path imageSpecsOutputFragment")
	}
}

func TestComposeQGenQuestion_setMode_homogeneousPlanRendersOnlyThatType(t *testing.T) {
	// All-MCQ set: no OE payload vocabulary anywhere.
	mcqOnly := ctxSet8mcq2oe()
	mcqOnly.TypePlan = []GenerationTypeQuota{{QuestionType: "mcq", Count: 5, MaxImages: 2}}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), mcqOnly)
	if strings.Contains(got, "oe_payload") || strings.Contains(got, "grader_tier") {
		t.Errorf("all-MCQ set leaked OE payload vocabulary")
	}
	if !strings.Contains(got, "\"question_type\": \"mcq\"") {
		t.Errorf("all-MCQ set missing the mcq candidate shape")
	}
	if !strings.Contains(got, "Across the 5 mcq questions, AT MOST 2 may include an image") {
		t.Errorf("all-MCQ set missing the mcq image cap")
	}

	// All-OE set: no MCQ-specific phrasing; the 0-image cap renders.
	oeOnly := ctxSet8mcq2oe()
	oeOnly.TypePlan = []GenerationTypeQuota{{QuestionType: "oe", Count: 3, MaxImages: 0}}
	got2 := ComposeQGenQuestionInstruction(StepGeneration3(), oeOnly)
	if !strings.Contains(got2, "oe_payload") {
		t.Errorf("all-OE set missing oe_payload")
	}
	if !strings.Contains(got2, "Across the 3 oe questions, AT MOST 0 may include an image") {
		t.Errorf("all-OE set missing the 0-image cap")
	}
}

func TestComposeQGenQuestion_setMode_isDeterministic(t *testing.T) {
	c := ctxSet8mcq2oe()
	// Two SEPARATE invocations (distinct expressions — avoids staticcheck SA4000)
	// must yield byte-identical prompts: set-mode composition is pure.
	first := ComposeQGenQuestionInstruction(StepGeneration3(), c)
	second := ComposeQGenQuestionInstruction(StepGeneration3(), c)
	if first != second {
		t.Errorf("set-mode composition is not deterministic (ADR-141 D2 break)")
	}
}

// -----------------------------------------------------------------------------
// (5) BuildTaskContextFromState — set-mode parsing + fail-loud.
// -----------------------------------------------------------------------------

func TestBuildTaskContextFromState_setModeFromJSONStringPlan(t *testing.T) {
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":           "tenant-x",
		"author_gcid":         "gcid-x",
		"intent":              "new_question",
		"set_mode":            true,
		"type_plan_json":      `[{"question_type":"mcq","count":8,"max_images":3},{"question_type":"oe","count":2,"max_images":1}]`,
		"avoid_concepts_json": `["evaporation","condensation"]`,
		"input_payload":       "the water cycle",
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if !tc.SetMode {
		t.Errorf("SetMode not read from state")
	}
	if len(tc.TypePlan) != 2 {
		t.Fatalf("TypePlan len = %d; want 2", len(tc.TypePlan))
	}
	if tc.TypePlan[0] != (GenerationTypeQuota{QuestionType: "mcq", Count: 8, MaxImages: 3}) {
		t.Errorf("TypePlan[0] = %+v", tc.TypePlan[0])
	}
	if tc.TypePlan[1] != (GenerationTypeQuota{QuestionType: "oe", Count: 2, MaxImages: 1}) {
		t.Errorf("TypePlan[1] = %+v", tc.TypePlan[1])
	}
	if len(tc.AvoidConcepts) != 2 || tc.AvoidConcepts[0] != "evaporation" || tc.AvoidConcepts[1] != "condensation" {
		t.Errorf("AvoidConcepts = %+v", tc.AvoidConcepts)
	}
}

func TestBuildTaskContextFromState_setModeFromNativePlan(t *testing.T) {
	// Defensive forward-compat: the executor stamps type_plan_json as a JSON
	// STRING (json.dumps) today, but readTypePlanFromState also tolerates a
	// native []any of maps (the shape author_options arrives in) by re-marshalling.
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":   "tenant-x",
		"author_gcid": "gcid-x",
		"set_mode":    true,
		"type_plan_json": []any{
			map[string]any{"question_type": "mcq", "count": float64(4), "max_images": float64(1)},
			map[string]any{"question_type": "oe", "count": float64(1), "max_images": float64(0)},
		},
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if len(tc.TypePlan) != 2 {
		t.Fatalf("TypePlan len = %d; want 2", len(tc.TypePlan))
	}
	if tc.TypePlan[0].QuestionType != "mcq" || tc.TypePlan[0].Count != 4 || tc.TypePlan[0].MaxImages != 1 {
		t.Errorf("native plan[0] mis-decoded: %+v", tc.TypePlan[0])
	}
	if tc.TypePlan[1].Count != 1 || tc.TypePlan[1].MaxImages != 0 {
		t.Errorf("native plan[1] mis-decoded: %+v", tc.TypePlan[1])
	}
}

func TestBuildTaskContextFromState_setModeEndToEndComposition(t *testing.T) {
	// State → context → prompt: the runtime path the live engine uses. A
	// set_mode state must resolve to the candidates+generation_summary wrapper.
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":      "tenant-x",
		"author_gcid":    "gcid-x",
		"set_mode":       true,
		"type_plan_json": `[{"question_type":"mcq","count":8,"max_images":3},{"question_type":"oe","count":2,"max_images":1}]`,
	}))
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	for _, want := range []string{
		"## [SET PLAN]",
		"Generate a SET of 10 questions",
		"\"candidates\"",
		"\"generation_summary\"",
		"generated_per_type",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("set-mode-from-state prompt missing %q", want)
		}
	}
}

func TestBuildTaskContextFromState_setModeFailLoud(t *testing.T) {
	cases := map[string]map[string]any{
		"empty plan": {
			"tenant_id": "t", "author_gcid": "g", "set_mode": true,
		},
		"invalid question_type": {
			"tenant_id": "t", "author_gcid": "g", "set_mode": true,
			"type_plan_json": `[{"question_type":"essay","count":3,"max_images":0}]`,
		},
		"malformed json": {
			"tenant_id": "t", "author_gcid": "g", "set_mode": true,
			"type_plan_json": `{ not json`,
		},
		"max_images exceeds count": {
			"tenant_id": "t", "author_gcid": "g", "set_mode": true,
			"type_plan_json": `[{"question_type":"mcq","count":2,"max_images":5}]`,
		},
		"negative count": {
			"tenant_id": "t", "author_gcid": "g", "set_mode": true,
			"type_plan_json": `[{"question_type":"mcq","count":-1,"max_images":0}]`,
		},
	}
	for name, st := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := BuildTaskContextFromState(newFakeState(st)); err == nil {
				t.Errorf("set_mode %s: expected a fail-loud error, got nil", name)
			}
		})
	}
}

func TestBuildTaskContextFromState_typePlanIgnoredWhenSetModeFalse(t *testing.T) {
	// set_mode=false: type_plan_json is NOT parsed or validated — even a
	// malformed one cannot break the single-candidate path.
	tc, err := BuildTaskContextFromState(newFakeState(map[string]any{
		"tenant_id":           "tenant-x",
		"author_gcid":         "gcid-x",
		"question_type":       "mcq",
		"type_plan_json":      `{ totally malformed`,
		"avoid_concepts_json": `["ignored"]`,
	}))
	if err != nil {
		t.Fatalf("single path must ignore a stray type_plan_json: %v", err)
	}
	if tc.SetMode {
		t.Errorf("SetMode should be false")
	}
	if tc.TypePlan != nil {
		t.Errorf("TypePlan should be nil in single mode, got %+v", tc.TypePlan)
	}
	if tc.AvoidConcepts != nil {
		t.Errorf("AvoidConcepts should be nil in single mode, got %+v", tc.AvoidConcepts)
	}
}

// -----------------------------------------------------------------------------
// firstDiff — readable byte-level diff for the golden + isolation assertions.
// -----------------------------------------------------------------------------

func firstDiff(want, got string) string {
	n := len(want)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		if want[i] != got[i] {
			lo := i - 40
			if lo < 0 {
				lo = 0
			}
			wHi, gHi := i+40, i+40
			if wHi > len(want) {
				wHi = len(want)
			}
			if gHi > len(got) {
				gHi = len(got)
			}
			return fmt.Sprintf("first diff at byte %d:\n  want …%q…\n   got …%q…", i, want[lo:wHi], got[lo:gHi])
		}
	}
	if len(want) != len(got) {
		return fmt.Sprintf("identical for the first %d bytes but lengths differ: want %d, got %d", n, len(want), len(got))
	}
	return "identical"
}
