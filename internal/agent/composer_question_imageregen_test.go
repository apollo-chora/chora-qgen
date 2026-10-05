// I1 (CHO-1822 redesign) — image_regen authoring intent + INTEGRAL forced images.
//
// Two changes, ONE composer source of truth for image rules:
//
//	(1) Item 1 — a FORCED image (single [IMAGE] block + the set [IMAGE BUDGET]
//	    forced branch) must instruct the question / model answer to REFERENCE +
//	    DEPEND ON the image (integral), not be a "comprehension aid ONLY".
//	(2) intent=image_regen — a focused single call that authors ONE image_spec
//	    {mode, source} from the CURRENT edited stem (+ model answer for the answer
//	    placement) + a refinement instruction, reusing the SAME mermaid/scene +
//	    integral rules. Routed through the qgen agent (no new role/deployment).
//
// Helpers ctxNewMCQWithImage + mapState live in composer_question_image_test.go.
package agent

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// (1) Item 1 — forced images are INTEGRAL (stem == answer, symmetric)
// -----------------------------------------------------------------------------

func TestImageGuidance_forcedStem_isIntegral(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQWithImage(true, false))
	lc := strings.ToLower(got)
	if !strings.Contains(lc, "integral") {
		t.Errorf("forced stem image must be INTEGRAL\n%s", got)
	}
	if !strings.Contains(lc, "reference") || !strings.Contains(lc, "depend") {
		t.Errorf("forced stem image must instruct the question to REFERENCE + DEPEND ON the image\n%s", got)
	}
	// The decorative / aid-only framing must be gone for forced images.
	if strings.Contains(lc, "comprehension aid only") || strings.Contains(lc, "must not alter the question") {
		t.Errorf("forced image must NOT be framed as comprehension-aid-only / must-not-alter\n%s", got)
	}
	// The no-answer-leak rule must survive.
	if !strings.Contains(lc, "not reveal") {
		t.Errorf("stem image must keep the no-answer-leak rule\n%s", got)
	}
}

func TestImageGuidance_forcedAnswer_isIntegral(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQWithImage(false, true))
	lc := strings.ToLower(got)
	if !strings.Contains(lc, "integral") || !strings.Contains(lc, "reference") {
		t.Errorf("forced answer image must be integral + referenced by the model answer\n%s", got)
	}
}

func TestImageBudget_forced_isIntegral(t *testing.T) {
	ctx := TaskContextQuestion{
		SetMode:  true,
		TypePlan: []GenerationTypeQuota{{QuestionType: "mcq", Count: 2, ImageForStem: true}},
	}
	got := imageBudgetBlock(ctx)
	lc := strings.ToLower(got)
	if !strings.Contains(lc, "integral") || !strings.Contains(lc, "reference") {
		t.Errorf("set-mode FORCED image budget must be integral\n%s", got)
	}
}

// A discretionary (AI-decide) budget stays supplementary — it must NOT force
// every question to reference an integral image.
func TestImageBudget_discretionary_notForcedIntegral(t *testing.T) {
	ctx := TaskContextQuestion{
		SetMode:  true,
		TypePlan: []GenerationTypeQuota{{QuestionType: "mcq", Count: 3, MaxImages: 1}},
	}
	got := imageBudgetBlock(ctx)
	if strings.Contains(strings.ToLower(got), "integral") {
		t.Errorf("discretionary budget must NOT force integral on every question\n%s", got)
	}
}

// -----------------------------------------------------------------------------
// (2) intent=image_regen — focused single image_spec authoring
// -----------------------------------------------------------------------------

func imageRegenState(placement string, extra map[string]any) *mapState {
	m := map[string]any{
		"tenant_id":     "t",
		"author_gcid":   "g",
		"intent":        "image_regen",
		"question_type": "mcq",
		"placement":     placement,
	}
	for k, v := range extra {
		m[k] = v
	}
	return &mapState{m: m}
}

func TestBuildTaskContextFromState_readsImageRegen(t *testing.T) {
	st := imageRegenState("answer", map[string]any{
		"refinement_prompt":    "rp",
		"current_stem":         "cs",
		"current_model_answer": "cma",
		"original_mode":        "scene",
		"original_source":      "os",
	})
	tc, err := BuildTaskContextFromState(st)
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.Intent != IntentImageRegen {
		t.Fatalf("intent not image_regen: %q", tc.Intent)
	}
	if tc.ImageRegen == nil {
		t.Fatalf("ImageRegen context nil")
	}
	r := tc.ImageRegen
	if r.Placement != ImagePlacementAnswer {
		t.Errorf("placement: got %q", r.Placement)
	}
	if r.RefinementPrompt != "rp" || r.CurrentStem != "cs" || r.CurrentModelAnswer != "cma" ||
		r.OriginalMode != "scene" || r.OriginalSource != "os" {
		t.Errorf("regen fields not read: %+v", r)
	}
}

func TestComposeImageRegen_stem(t *testing.T) {
	st := imageRegenState("stem", map[string]any{
		"refinement_prompt": "make the cycle clearer",
		"current_stem":      "Which stage follows the tadpole in the frog life cycle?",
		"original_mode":     "mermaid",
		"original_source":   "flowchart LR; egg-->tadpole;",
	})
	tc, err := BuildTaskContextFromState(st)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	lc := strings.ToLower(got)
	if !strings.Contains(lc, "mermaid") || !strings.Contains(lc, "scene") {
		t.Errorf("image_regen must let the model choose mermaid vs scene\n%s", got)
	}
	if !strings.Contains(got, "Which stage follows the tadpole in the frog life cycle?") {
		t.Errorf("image_regen stem must include the current stem\n%s", got)
	}
	if !strings.Contains(got, "make the cycle clearer") {
		t.Errorf("image_regen must include the refinement instruction\n%s", got)
	}
	if !strings.Contains(got, "image_spec") {
		t.Errorf("image_regen must ask for an image_spec\n%s", got)
	}
	if !strings.Contains(lc, "integral") {
		t.Errorf("image_regen image must be integral to the question\n%s", got)
	}
	if !strings.Contains(got, "flowchart LR; egg-->tadpole;") {
		t.Errorf("image_regen should offer the original source to refine\n%s", got)
	}
}

func TestComposeImageRegen_answerIncludesModelAnswer(t *testing.T) {
	st := imageRegenState("answer", map[string]any{
		"refinement_prompt":    "show each step",
		"current_stem":         "Explain photosynthesis.",
		"current_model_answer": "Plants convert light energy into glucose via the Calvin cycle.",
	})
	tc, _ := BuildTaskContextFromState(st)
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if !strings.Contains(got, "Plants convert light energy into glucose via the Calvin cycle.") {
		t.Errorf("image_regen answer must include the current model answer\n%s", got)
	}
}

// A STEM regen must NOT leak the model answer into the prompt context.
func TestComposeImageRegen_stemOmitsModelAnswer(t *testing.T) {
	st := imageRegenState("stem", map[string]any{
		"refinement_prompt":    "clearer",
		"current_stem":         "Stem here",
		"current_model_answer": "SECRET-ANSWER-LEAK",
	})
	tc, _ := BuildTaskContextFromState(st)
	got := ComposeQGenQuestionInstruction(StepGeneration3(), tc)
	if strings.Contains(got, "SECRET-ANSWER-LEAK") {
		t.Errorf("stem image_regen must NOT include the model answer (leak)\n%s", got)
	}
}
