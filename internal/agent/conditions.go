package agent

import "strconv"

// conditions.go — ADR-197 M-A condition extractors for the qgen crews.
//
// These surface the prompt-shaping DISCRIMINANTS (intent, question_type, hints,
// image opt-ins, set-mode, critic attempt) — NOT the author's content/PII — so
// O+ can render "which conditions produced this decision". They are pure,
// best-effort (return nil if the state can't form a valid task context, in
// which case the InstructionProvider would already have failed the turn), and
// reuse Build*TaskContextFromState so they can never drift from what actually
// drives the composed prompt.
//
// The parameter is the qgen-local stateGetter (Get-only); session.ReadonlyState
// satisfies it, so a main wraps QuestionConditions/CriticConditions into a
// promptstamping.ConditionExtractor.

// QuestionConditions extracts the qgen_question prompt discriminants.
func QuestionConditions(state stateGetter) map[string]string {
	tc, err := BuildTaskContextFromState(state)
	if err != nil {
		return nil
	}
	c := map[string]string{"intent": string(tc.Intent)}
	if tc.SetMode {
		// In set-mode the per-type plan is the source of truth; the single
		// question_type is moot, so surface the mode instead.
		c["set_mode"] = "true"
	} else {
		c["question_type"] = string(tc.QuestionType)
	}
	if tc.SubjectHint != "" {
		c["subject_hint"] = tc.SubjectHint
	}
	if tc.CognitiveLevelHint != "" {
		c["cognitive_level_hint"] = tc.CognitiveLevelHint
	}
	if tc.DifficultyHint != "" {
		c["difficulty_hint"] = tc.DifficultyHint
	}
	if tc.ImageForStem {
		c["image_for_stem"] = "true"
	}
	if tc.ImageForAnswer {
		c["image_for_answer"] = "true"
	}
	if tc.ImageRegen != nil && tc.ImageRegen.Placement != "" {
		c["image_regen_placement"] = string(tc.ImageRegen.Placement)
	}
	return c
}

// CriticConditions extracts the qgen_critic prompt discriminants.
func CriticConditions(state stateGetter) map[string]string {
	tc, err := BuildCriticTaskContextFromState(state)
	if err != nil {
		return nil
	}
	c := map[string]string{
		"attempt_index": strconv.Itoa(tc.AttemptIndex),
	}
	if tc.SetMode {
		// In set mode the candidates self-declare question_type; the single
		// field is moot, so surface the mode instead (mirrors
		// QuestionConditions).
		c["set_mode"] = "true"
	} else {
		c["question_type"] = string(tc.QuestionType)
	}
	if tc.PriorCriticNotes != "" {
		c["has_prior_notes"] = "true"
	}
	return c
}
