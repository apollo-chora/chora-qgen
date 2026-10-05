package agent

// critic_golden_test.go - frozen composed-prompt goldens for the qgen_critic
// agent (CHO-2368, ADR-197 catalogue). Mirrors the qgen_question golden
// discipline in composer_question_set_test.go: the golden files are the
// byte-pinned rendering of the CURRENT embedded blocks with NO overrides, and
// the ADR-197 baseline seed generator (orchestrator
// domain/prompt_registry/baseline_seedspec.py) slices its catalogue segment
// bodies out of these files - so a composer edit that changes the prompt
// either regenerates the golden (visible in review + REDs the seed drift
// test) or fails here.
//
// Regenerate with:
//
//	go test ./internal/agent/ -run CriticGolden -update

import (
	"os"
	"path/filepath"
	"testing"
)

// criticGoldenCtx returns the canonical no-override CriticTaskContext used to
// freeze the composed prompt. Fixture IDs mirror the qgen_question goldens.
func criticGoldenCtx(qt QuestionType) CriticTaskContext {
	candidate := `{"stem": "Which ceremony closes a Scrum sprint?", "question_type": "mcq"}`
	if qt == QuestionTypeOE {
		candidate = `{"stem": "Explain how a Scrum sprint review differs from a retrospective.", "question_type": "oe"}`
	}
	return CriticTaskContext{
		TenantID:      "01957c8c-0000-7000-8888-000088880000",
		AuthorGCID:    "01957c8c-0000-7000-9999-000099990000",
		JobID:         "01957c8c-9999-7000-7777-9999aaaa9999",
		QuestionType:  qt,
		AttemptIndex:  0,
		MaxAttempts:   4,
		AuthorPrompt:  "Generate a question on Scrum sprints",
		CandidateJSON: candidate,
	}
}

func TestComposeCritic_CriticGoldenByteIdentical(t *testing.T) {
	cases := map[string]QuestionType{
		"mcq": QuestionTypeMCQ,
		"oe":  QuestionTypeOE,
	}
	for name, qt := range cases {
		t.Run(name, func(t *testing.T) {
			got := ComposeCriticInstruction(CriticStep(), criticGoldenCtx(qt))
			path := filepath.Join("testdata", "golden_critic_"+name+".txt")
			if *updateGolden {
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatalf("write golden %s: %v", path, err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read golden %s (run with -update to generate): %v", path, err)
			}
			if got != string(want) {
				t.Fatalf("composed critic prompt drifted from %s - if intentional, regenerate with -update AND expect the ADR-197 baseline seed drift test to demand a catalogue update", path)
			}
		})
	}
}
