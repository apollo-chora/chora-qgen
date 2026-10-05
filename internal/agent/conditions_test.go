package agent

// RED-first tests for the ADR-197 M-A condition extractors. These surface the
// prompt-shaping discriminants (NOT the author's content/PII) so O+ can render
// "which conditions produced this decision". They mirror BuildTaskContextFromState
// / BuildCriticTaskContextFromState so they can never drift from what actually
// drives the prompt. Reuses the in-package mapState test fake.

import "testing"

func TestQuestionConditions_newMCQ(t *testing.T) {
	st := &mapState{m: map[string]any{"intent": "new_question", "question_type": "mcq"}}
	c := QuestionConditions(st)
	if c["intent"] != "new_question" {
		t.Errorf("intent: got %q", c["intent"])
	}
	if c["question_type"] != "mcq" {
		t.Errorf("question_type: got %q", c["question_type"])
	}
}

func TestQuestionConditions_includesHintsAndImage(t *testing.T) {
	st := &mapState{m: map[string]any{
		"intent":          "new_question",
		"question_type":   "oe",
		"difficulty_hint": "hard",
		"subject_hint":    "biology",
		"image_for_stem":  true,
	}}
	c := QuestionConditions(st)
	if c["difficulty_hint"] != "hard" {
		t.Errorf("difficulty_hint: got %q", c["difficulty_hint"])
	}
	if c["subject_hint"] != "biology" {
		t.Errorf("subject_hint: got %q", c["subject_hint"])
	}
	if c["image_for_stem"] != "true" {
		t.Errorf("image_for_stem: got %q", c["image_for_stem"])
	}
}

func TestQuestionConditions_omitsBlankHints(t *testing.T) {
	st := &mapState{m: map[string]any{"intent": "new_question", "question_type": "mcq"}}
	c := QuestionConditions(st)
	if _, ok := c["difficulty_hint"]; ok {
		t.Error("blank difficulty_hint must be omitted")
	}
	if _, ok := c["image_for_stem"]; ok {
		t.Error("false image_for_stem must be omitted")
	}
}

func TestQuestionConditions_invalidStateReturnsNil(t *testing.T) {
	// Invalid question_type (non-set-mode) makes BuildTaskContextFromState fail
	// loud; the extractor is best-effort and returns nil rather than a partial map.
	st := &mapState{m: map[string]any{"intent": "new_question", "question_type": "bogus"}}
	if c := QuestionConditions(st); c != nil {
		t.Errorf("invalid state must yield nil; got %v", c)
	}
}

func TestQuestionConditions_setModeReplacesQuestionType(t *testing.T) {
	// In set mode the per-type plan is the source of truth, so the single
	// question_type is moot. O+ must see set_mode=true INSTEAD of a
	// question_type that would misdescribe a mixed-type batch.
	st := &mapState{m: map[string]any{
		"set_mode":       true,
		"question_type":  "mixed",
		"type_plan_json": `[{"question_type":"mcq","count":2,"max_images":1},{"question_type":"oe","count":1,"max_images":0}]`,
	}}
	c := QuestionConditions(st)
	if c["set_mode"] != "true" {
		t.Errorf("set_mode: got %q, want \"true\"", c["set_mode"])
	}
	if _, ok := c["question_type"]; ok {
		t.Errorf("set mode must NOT surface a single question_type; got %q", c["question_type"])
	}
}

func TestQuestionConditions_includesCognitiveLevelHint(t *testing.T) {
	// The Bloom level shapes the generated stem, so it is a discriminant an
	// auditor needs, unlike the author's free-form prompt which is content.
	st := &mapState{m: map[string]any{
		"intent":               "new_question",
		"question_type":        "mcq",
		"cognitive_level_hint": "analysis",
	}}
	c := QuestionConditions(st)
	if c["cognitive_level_hint"] != "analysis" {
		t.Errorf("cognitive_level_hint: got %q", c["cognitive_level_hint"])
	}
}

func TestQuestionConditions_includesImageForAnswer(t *testing.T) {
	// image_for_answer is an independent author opt-in from image_for_stem: the
	// answer flag alone must surface, and must not be reported as a stem image.
	st := &mapState{m: map[string]any{
		"intent":           "new_question",
		"question_type":    "oe",
		"image_for_answer": true,
	}}
	c := QuestionConditions(st)
	if c["image_for_answer"] != "true" {
		t.Errorf("image_for_answer: got %q", c["image_for_answer"])
	}
	if _, ok := c["image_for_stem"]; ok {
		t.Error("image_for_stem must stay absent when only the answer was opted in")
	}
}

func TestQuestionConditions_imageRegenPlacement(t *testing.T) {
	// intent=image_regen re-authors ONE illustration. Which part it targets is
	// the whole discriminant of that turn, so the placement rides the map.
	st := &mapState{m: map[string]any{
		"intent":            "image_regen",
		"question_type":     "mcq",
		"placement":         "answer",
		"refinement_prompt": "make the diagram simpler",
	}}
	c := QuestionConditions(st)
	if c["intent"] != "image_regen" {
		t.Errorf("intent: got %q", c["intent"])
	}
	if c["image_regen_placement"] != "answer" {
		t.Errorf("image_regen_placement: got %q, want \"answer\"", c["image_regen_placement"])
	}
}

func TestQuestionConditions_imageRegenWithoutPlacementOmitsKey(t *testing.T) {
	// A regen turn that never carried a placement has nothing to report; the
	// extractor must omit the key rather than stamp an empty-string condition.
	st := &mapState{m: map[string]any{
		"intent":        "image_regen",
		"question_type": "mcq",
	}}
	c := QuestionConditions(st)
	if _, ok := c["image_regen_placement"]; ok {
		t.Errorf("blank placement must be omitted; got %q", c["image_regen_placement"])
	}
}

func TestCriticConditions_basic(t *testing.T) {
	st := &mapState{m: map[string]any{
		"input_payload": `{"stem":"x"}`,
		"question_type": "oe",
		"attempt_index": float64(2),
	}}
	c := CriticConditions(st)
	if c["question_type"] != "oe" {
		t.Errorf("question_type: got %q", c["question_type"])
	}
	if c["attempt_index"] != "2" {
		t.Errorf("attempt_index: got %q", c["attempt_index"])
	}
}

func TestCriticConditions_priorNotesFlag(t *testing.T) {
	st := &mapState{m: map[string]any{
		"input_payload":      `{"stem":"x"}`,
		"question_type":      "mcq",
		"prior_critic_notes": "fix distractors",
	}}
	c := CriticConditions(st)
	if c["has_prior_notes"] != "true" {
		t.Errorf("has_prior_notes: got %q", c["has_prior_notes"])
	}
}

func TestCriticConditions_missingCandidateReturnsNil(t *testing.T) {
	st := &mapState{m: map[string]any{"question_type": "mcq"}}
	if c := CriticConditions(st); c != nil {
		t.Errorf("missing candidate must yield nil; got %v", c)
	}
}
