package agent

import (
	"strings"
	"testing"
)

// ComposeQGenInstruction must emit the same six CREATE blocks as the
// Familiar composer (`internal/agent/builder.go` in familiar_adk_go), but
// tailored per-step. QGen has 6 sub-agents (Validator / Classifier /
// Web Researcher / QA Generator / Evaluator / Reporter) each with its own
// [ROLE] + [TASK] + [EXPECTED OUTPUT].
//
// Per ADR-141 D2 transparency: deterministic — same (step, ctx) in always
// yields same prompt out.

func TestComposeQGenInstruction_emitsSixCreateBlocksInOrder(t *testing.T) {
	for _, step := range AllSteps() {
		got := ComposeQGenInstruction(step, TaskContext{
			TenantID:        "01957c8c-0000-7000-8888-000088880000",
			BatchID:         "01957c8c-9999-7000-7777-9999aaaa9999",
			SubjectHint:     "agile-estimation",
			DifficultyHint:  "intermediate",
			DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE",
		})
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
				t.Errorf("step %s: CREATE block %q missing\n--- prompt:\n%s", step.Name, b, got)
				continue
			}
			if i <= lastIdx {
				t.Errorf("step %s: CREATE blocks out of order; %q at %d after %d",
					step.Name, b, i, lastIdx)
			}
			lastIdx = i
		}
	}
}

func TestComposeQGenInstruction_isDeterministic(t *testing.T) {
	ctx := TaskContext{
		TenantID:        "tenant-stub",
		BatchID:         "batch-stub",
		SubjectHint:     "kotlin-coroutines",
		DifficultyHint:  "advanced",
		DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE",
	}
	for _, step := range AllSteps() {
		a := ComposeQGenInstruction(step, ctx)
		b := ComposeQGenInstruction(step, ctx)
		if a != b {
			t.Errorf("step %s: ComposeQGenInstruction not deterministic (IMDA D2 break)", step.Name)
		}
	}
}

func TestComposeQGenInstruction_distinguishesSteps(t *testing.T) {
	ctx := TaskContext{TenantID: "t", BatchID: "b", SubjectHint: "s", DifficultyHint: "intermediate", DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE"}
	prompts := make(map[string]string, 6)
	for _, step := range AllSteps() {
		prompts[step.Name] = ComposeQGenInstruction(step, ctx)
	}
	for n1, p1 := range prompts {
		for n2, p2 := range prompts {
			if n1 != n2 && p1 == p2 {
				t.Errorf("steps %s and %s produced identical instructions", n1, n2)
			}
		}
	}
}

func TestComposeQGenInstruction_contextFieldsSurface(t *testing.T) {
	ctx := TaskContext{
		TenantID:        "tenant-xyz",
		BatchID:         "batch-abc",
		SubjectHint:     "graph-algorithms",
		DifficultyHint:  "advanced",
		DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE",
	}
	got := ComposeQGenInstruction(StepAssurance(), ctx)
	for _, want := range []string{"tenant-xyz", "graph-algorithms", "advanced"} {
		if !strings.Contains(got, want) {
			t.Errorf("CONTEXT block missing %q; prompt:\n%s", want, got)
		}
	}
}

func TestAllSteps_returnsSixSteps_5CanonicalPlusConditionalWebResearcher(t *testing.T) {
	// 5 canonical reusable roles per crew-composition SKILL §3 + 1
	// conditional WebResearcher per §3.1 conditional-agent pattern.
	steps := AllSteps()
	if len(steps) != 6 {
		t.Fatalf("AllSteps() must return 6 steps; got %d", len(steps))
	}
	want := []string{
		"qgen_assurance",
		"qgen_fitness",
		"qgen_web_researcher",
		"qgen_generation",
		"qgen_evaluation",
		"qgen_delivery",
	}
	for i, w := range want {
		if steps[i].Name != w {
			t.Errorf("step %d: want %q; got %q", i, w, steps[i].Name)
		}
	}
}

func TestStepConstructors_haveCanonicalNames(t *testing.T) {
	pairs := []struct {
		got, want string
	}{
		{StepAssurance().Name, "qgen_assurance"},
		{StepFitness().Name, "qgen_fitness"},
		{StepWebResearcher().Name, "qgen_web_researcher"},
		{StepGeneration().Name, "qgen_generation"},
		{StepEvaluation().Name, "qgen_evaluation"},
		{StepDelivery().Name, "qgen_delivery"},
	}
	for _, p := range pairs {
		if p.got != p.want {
			t.Errorf("step name: got %q want %q", p.got, p.want)
		}
	}
}

func TestStepWebResearcher_carriesConditionalSkipRule(t *testing.T) {
	// Per crew-composition SKILL §3.1: the WebResearcher is the only
	// non-canonical (conditional) agent in the QGen pipeline. Its prompt
	// MUST advertise the conditional skip rule so the runtime LLM can
	// emit the cheap no-op return when Fitness verdict says
	// needs_web_research=false.
	step := StepWebResearcher()
	ctx := TaskContext{TenantID: "t", BatchID: "b", SubjectHint: "s", DifficultyHint: "intermediate", DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE"}
	got := ComposeQGenInstruction(step, ctx)

	requiredCues := []string{
		"CONDITIONAL SKIP RULE",
		"needs_web_research=false",
		"high_confidence_classification",
		"invoked",
	}
	for _, cue := range requiredCues {
		if !strings.Contains(got, cue) {
			t.Errorf("WebResearcher prompt missing conditional cue %q", cue)
		}
	}
}

func TestStepFitness_signalsConditionalNeedsWebResearch(t *testing.T) {
	// Fitness must emit needs_web_research bool so the conditional
	// WebResearcher can self-skip. Drift breaks the conditional pattern.
	step := StepFitness()
	ctx := TaskContext{TenantID: "t", BatchID: "b", SubjectHint: "s", DifficultyHint: "intermediate", DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE"}
	got := ComposeQGenInstruction(step, ctx)

	if !strings.Contains(got, "needs_web_research") {
		t.Errorf("Fitness prompt must mention needs_web_research field (drives the conditional WebResearcher)")
	}
}
