// W8 — author-opt-in image_specs tests for the qgen_question generation
// composer.
//
// Per CR §5.3 + the 2026-06-01 coordinator refinement, image generation is
// AUTHOR-OPT-IN, not model autonomy: the author selects (per QUESTION part
// and/or MODEL-ANSWER part) whether an illustration is wanted via two input
// bools — image_for_stem + image_for_answer (default false). The generation
// agent emits a LIST `image_specs: [{mode, source, placement}, ...]` (0-2
// entries) on the candidate:
//
//   - image_for_stem=true   → a spec with placement="stem"
//   - image_for_answer=true → a spec with placement="answer"
//   - both true             → TWO specs
//   - both false (default)  → NOTHING (unchanged behaviour)
//
// The MODEL still chooses `mode` per spec (mermaid for a structural diagram
// vs scene for an illustrative image PROMPT) and authors `source`; WHETHER +
// WHICH PART is the author's choice via the flags.
//
// These tests cover: (1) the ImageSpec struct round-trips mermaid + scene
// through JSON; (2) the four opt-in cases (stem-only / answer-only / both /
// neither) drive the right prompt guidance; (3) the default (neither) path is
// byte-compatible with today's candidate-shape prompt.
package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// (1) ImageSpec JSON round-trip — mermaid + scene
// -----------------------------------------------------------------------------

func TestImageSpec_RoundTripsMermaid(t *testing.T) {
	in := ImageSpec{
		Mode:      ImageModeMermaid,
		Source:    "graph TD; A[Start]-->B[Process]-->C[End];",
		Placement: ImagePlacementStem,
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out ImageSpec
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out != in {
		t.Errorf("mermaid round-trip mismatch: got %+v, want %+v", out, in)
	}
	// Wire keys MUST be the snake_case the Python _map_response passes through
	// opaquely + the FE render_image node reads.
	for _, k := range []string{`"mode"`, `"source"`, `"placement"`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("marshalled image_spec missing key %s: %s", k, b)
		}
	}
	if !strings.Contains(string(b), `"mermaid"`) {
		t.Errorf("mermaid mode did not serialise to \"mermaid\": %s", b)
	}
}

func TestImageSpec_RoundTripsScene(t *testing.T) {
	in := ImageSpec{
		Mode:      ImageModeScene,
		Source:    "A photorealistic cross-section of a plant leaf showing chloroplasts.",
		Placement: ImagePlacementAnswer,
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out ImageSpec
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out != in {
		t.Errorf("scene round-trip mismatch: got %+v, want %+v", out, in)
	}
	if !strings.Contains(string(b), `"scene"`) {
		t.Errorf("scene mode did not serialise to \"scene\": %s", b)
	}
	if !strings.Contains(string(b), `"answer"`) {
		t.Errorf("answer placement did not serialise to \"answer\": %s", b)
	}
}

// A list of image_specs (0-2 entries) round-trips — the on-candidate shape.
func TestImageSpecs_ListRoundTrips(t *testing.T) {
	in := []ImageSpec{
		{Mode: ImageModeMermaid, Source: "graph LR; A-->B;", Placement: ImagePlacementStem},
		{Mode: ImageModeScene, Source: "An illustration of B.", Placement: ImagePlacementAnswer},
	}
	b, err := json.Marshal(map[string]any{"image_specs": in})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var wrap struct {
		ImageSpecs []ImageSpec `json:"image_specs"`
	}
	if err := json.Unmarshal(b, &wrap); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(wrap.ImageSpecs) != 2 || wrap.ImageSpecs[0] != in[0] || wrap.ImageSpecs[1] != in[1] {
		t.Errorf("image_specs list round-trip mismatch: got %+v, want %+v", wrap.ImageSpecs, in)
	}
}

// -----------------------------------------------------------------------------
// (2) Four author-opt-in cases drive the right generation prompt guidance.
//
// The composer must:
//   - mention image_specs + the author's opted-in placement(s) when a flag is set
//   - tell the model to choose mermaid (structure) vs scene (illustration)
//   - instruct OMIT when neither flag is set
// -----------------------------------------------------------------------------

func ctxNewMCQWithImage(stem, answer bool) TaskContextQuestion {
	c := ctxNewMCQ()
	c.ImageForStem = stem
	c.ImageForAnswer = answer
	return c
}

func TestComposeQGenQuestion_imageStemOnly(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQWithImage(true, false))
	if !strings.Contains(got, "image_specs") {
		t.Errorf("stem-only: prompt missing image_specs guidance")
	}
	if !strings.Contains(got, `placement="stem"`) && !strings.Contains(got, "placement=\"stem\"") {
		t.Errorf("stem-only: prompt does not direct a stem-placement spec\n%s", got)
	}
	// Must NOT instruct an answer-placement spec when only the stem is opted in.
	if strings.Contains(got, `placement="answer"`) {
		t.Errorf("stem-only: prompt wrongly directs an answer-placement spec")
	}
	// Model still chooses mermaid vs scene.
	if !strings.Contains(strings.ToLower(got), "mermaid") || !strings.Contains(strings.ToLower(got), "scene") {
		t.Errorf("stem-only: prompt does not let the model choose mermaid vs scene")
	}
}

func TestComposeQGenQuestion_imageAnswerOnly(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQWithImage(false, true))
	if !strings.Contains(got, "image_specs") {
		t.Errorf("answer-only: prompt missing image_specs guidance")
	}
	if !strings.Contains(got, `placement="answer"`) {
		t.Errorf("answer-only: prompt does not direct an answer-placement spec\n%s", got)
	}
	if strings.Contains(got, `placement="stem"`) {
		t.Errorf("answer-only: prompt wrongly directs a stem-placement spec")
	}
}

func TestComposeQGenQuestion_imageBoth(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQWithImage(true, true))
	if !strings.Contains(got, `placement="stem"`) || !strings.Contains(got, `placement="answer"`) {
		t.Errorf("both: prompt must direct BOTH a stem + an answer spec\n%s", got)
	}
	// Should signal two specs are expected.
	if !strings.Contains(got, "image_specs") {
		t.Errorf("both: prompt missing image_specs guidance")
	}
}

func TestComposeQGenQuestion_imageNeitherOmits(t *testing.T) {
	got := ComposeQGenQuestionInstruction(StepGeneration3(), ctxNewMCQWithImage(false, false))
	// With neither flag set the prompt MUST NOT solicit image_specs at all —
	// default behaviour is unchanged (no image guidance leaks in).
	if strings.Contains(got, "image_specs") {
		t.Errorf("neither: prompt must NOT mention image_specs when no flag is set\n%s", got)
	}
	if strings.Contains(got, `placement="stem"`) || strings.Contains(got, `placement="answer"`) {
		t.Errorf("neither: prompt must NOT direct any placement when no flag is set")
	}
}

// -----------------------------------------------------------------------------
// (3) Default (neither flag) path is byte-compatible with today's prompt.
//
// The image opt-in is strictly additive: a TaskContextQuestion with both flags
// false MUST compose the EXACT SAME string as the pre-W8 composer for the same
// inputs. We assert this by composing the canonical ctxNewMCQ() (flags default
// false) and confirming no image vocabulary appears anywhere.
// -----------------------------------------------------------------------------

func TestComposeQGenQuestion_defaultPathHasNoImageVocabulary(t *testing.T) {
	for _, c := range []TaskContextQuestion{ctxNewMCQ(), ctxNewOE(), ctxFillMCQ(), ctxFillOE()} {
		got := ComposeQGenQuestionInstruction(StepGeneration3(), c)
		for _, banned := range []string{"image_specs", "image_spec", "mermaid", `placement="stem"`, `placement="answer"`} {
			if strings.Contains(got, banned) {
				t.Errorf("default path (no opt-in) leaked image vocabulary %q for intent=%s qt=%s",
					banned, c.Intent, c.QuestionType)
			}
		}
	}
}

// -----------------------------------------------------------------------------
// (2b) Flags read from session state (the runtime path the live engine uses).
// -----------------------------------------------------------------------------

func TestBuildTaskContextFromState_readsImageFlags(t *testing.T) {
	st := &mapState{m: map[string]any{
		"tenant_id":        "tenant-x",
		"author_gcid":      "gcid-x",
		"intent":           "new_question",
		"question_type":    "mcq",
		"input_payload":    "Generate an MCQ on the water cycle",
		"image_for_stem":   true,
		"image_for_answer": false,
	}}
	tc, err := BuildTaskContextFromState(st)
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if !tc.ImageForStem {
		t.Errorf("ImageForStem not read from state")
	}
	if tc.ImageForAnswer {
		t.Errorf("ImageForAnswer should default false when state says false")
	}
}

func TestBuildTaskContextFromState_imageFlagsDefaultFalseWhenAbsent(t *testing.T) {
	st := &mapState{m: map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"intent":        "new_question",
		"question_type": "mcq",
	}}
	tc, err := BuildTaskContextFromState(st)
	if err != nil {
		t.Fatalf("BuildTaskContextFromState: %v", err)
	}
	if tc.ImageForStem || tc.ImageForAnswer {
		t.Errorf("image flags should default false when absent: stem=%v answer=%v", tc.ImageForStem, tc.ImageForAnswer)
	}
}

// mapState is a minimal stateGetter for the runtime-path tests.
type mapState struct{ m map[string]any }

func (s *mapState) Get(k string) (any, error) {
	v, ok := s.m[k]
	if !ok {
		return nil, errNotFound
	}
	return v, nil
}

var errNotFound = &stateKeyMissing{}

type stateKeyMissing struct{}

func (*stateKeyMissing) Error() string { return "key not found" }
