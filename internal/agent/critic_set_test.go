// Set-mode (batch critique) composer tests for the qgen_critic agent (CHO-2397,
// ADR-251 D2/D3).
//
// Set mode flips the critic from critiquing ONE candidate to critiquing a SET
// of candidates in one call, returning one verdict per candidate under the
// {"verdicts": [...]} wrapper with a strict candidate_id echo contract. The
// hard contract is that the LIVE single-candidate path (SetMode=false) is
// byte-for-byte UNCHANGED (proven by critic_golden_test.go against the frozen
// goldens, plus the stray-key inertness check below).
//
// Coverage mirrors composer_question_set_test.go:
//   - BuildCriticTaskContextFromState set-mode parsing: flag read, single
//     question_type validation skipped (candidates self-declare), the
//     input_payload fail-loud guard preserved, stray set_mode=false inert.
//   - Full ComposeCriticInstruction in set mode: plural candidates block, both
//     per-type rubrics embedded, echo contract stated in [TASK] and enforced in
//     [EXPECTED OUTPUT], set-shaped examples + audience, single-shape blocks
//     absent.
//   - ADR-197 override routing is uniform across modes: role/examples/task
//     overridable, the [EXPECTED OUTPUT] contract never.
//   - CriticConditions surfaces set_mode instead of the moot question_type.
//   - Golden byte-pin of the composed set prompt (testdata/golden_critic_set.txt).
package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// criticSetState returns the canonical set-mode session state: the flag plus a
// 2-candidate mixed payload under input_payload (the same state key the single
// path reads; the orchestrator stamps the candidates array there in set mode).
func criticSetState() *fakeState {
	return &fakeState{data: map[string]any{
		"tenant_id":     "tenant-x",
		"author_gcid":   "gcid-x",
		"job_id":        "job-x",
		"set_mode":      true,
		"input_payload": `{"candidates": [{"candidate_id": "c0", "question_type": "mcq", "stem": "s"}, {"candidate_id": "c1", "question_type": "oe", "stem": "s2"}]}`,
	}}
}

// criticSetGoldenCtx is the canonical no-override set-mode context used to
// freeze the composed set prompt. Fixture IDs mirror criticGoldenCtx.
func criticSetGoldenCtx() CriticTaskContext {
	return CriticTaskContext{
		TenantID:     "01957c8c-0000-7000-8888-000088880000",
		AuthorGCID:   "01957c8c-0000-7000-9999-000099990000",
		JobID:        "01957c8c-9999-7000-7777-9999aaaa9999",
		QuestionType: QuestionTypeMCQ, // moot in set mode (mirrors the generator)
		SetMode:      true,
		AttemptIndex: 0,
		MaxAttempts:  4,
		AuthorPrompt: "Generate a mixed set on Scrum sprints",
		CandidateJSON: `{"candidates": [{"candidate_id": "c0", "question_type": "mcq", ` +
			`"stem": "Which ceremony closes a Scrum sprint?"}, {"candidate_id": "c1", ` +
			`"question_type": "oe", "stem": "Explain how a sprint review differs from a retrospective."}]}`,
	}
}

// -----------------------------------------------------------------------------
// BuildCriticTaskContextFromState: set-mode parsing
// -----------------------------------------------------------------------------

func TestBuildCriticTaskContextFromState_setModeFlag(t *testing.T) {
	tc, err := BuildCriticTaskContextFromState(criticSetState())
	if err != nil {
		t.Fatalf("set-mode state must build: %v", err)
	}
	if !tc.SetMode {
		t.Fatalf("SetMode must be true when state carries set_mode=true")
	}
	if !strings.Contains(tc.CandidateJSON, `"candidate_id": "c0"`) {
		t.Fatalf("CandidateJSON must carry the candidates payload, got %q", tc.CandidateJSON)
	}
}

func TestBuildCriticTaskContextFromState_setModeSkipsSingleTypeValidation(t *testing.T) {
	// Candidates self-declare question_type in set mode; the single field is
	// moot (the orchestrator may stamp "mixed" or omit it), mirroring the
	// generator's set-mode contract.
	st := criticSetState()
	st.data["question_type"] = "mixed"
	tc, err := BuildCriticTaskContextFromState(st)
	if err != nil {
		t.Fatalf("set mode must not validate the moot single question_type: %v", err)
	}
	if !tc.SetMode {
		t.Fatalf("SetMode must be true")
	}
}

func TestBuildCriticTaskContextFromState_setModeStillRequiresPayload(t *testing.T) {
	st := criticSetState()
	st.data["input_payload"] = "   "
	if _, err := BuildCriticTaskContextFromState(st); err == nil {
		t.Fatalf("set mode with an empty input_payload must fail loud")
	}
}

func TestBuildCriticTaskContextFromState_straySetModeFalseInert(t *testing.T) {
	// A stray set_mode=false key must leave the single path untouched,
	// including its fail-loud question_type validation.
	st := &fakeState{data: map[string]any{
		"set_mode":      false,
		"question_type": "bogus",
		"input_payload": `{"stem": "x"}`,
	}}
	if _, err := BuildCriticTaskContextFromState(st); err == nil {
		t.Fatalf("single path must still refuse an invalid question_type")
	}
}

// -----------------------------------------------------------------------------
// ComposeCriticInstruction: set-mode blocks
// -----------------------------------------------------------------------------

func TestComposeCritic_setMode_candidatesBlockAndRubrics(t *testing.T) {
	got := ComposeCriticInstruction(CriticStep(), criticSetGoldenCtx())

	for _, want := range []string{
		"## [CANDIDATES TO CRITIQUE]",
		criticSetGoldenCtx().CandidateJSON, // embedded verbatim
		"ECHO CONTRACT",
		"exactly once",
		"H1. **Option count**",            // MCQ rubric embedded
		"H1. **Model answer word count**", // OE rubric embedded
		`{"verdicts":`,
		`"candidate_id"`,
		"Example set critique",        // set-shaped examples
		"parse_critique_set_response", // set-shaped audience
		`{"verdicts": []}`,            // whole-input-unparseable rule
	} {
		if !strings.Contains(got, want) {
			t.Errorf("set-mode prompt missing %q", want)
		}
	}

	for _, absent := range []string{
		"## [CANDIDATE TO CRITIQUE]",              // singular block heading
		"Example MCQ acceptance:",                 // single-shape examples
		"If you cannot parse the input candidate", // single-shape unparseable tail
	} {
		if strings.Contains(got, absent) {
			t.Errorf("set-mode prompt must not contain single-shape fragment %q", absent)
		}
	}
}

func TestComposeCritic_setMode_verdictPerCandidateRules(t *testing.T) {
	got := ComposeCriticInstruction(CriticStep(), criticSetGoldenCtx())
	// The per-candidate unparseable rule keeps one bad candidate from
	// suppressing the others' verdicts; the echo contract is restated in the
	// non-overridable [EXPECTED OUTPUT] so it survives any task override.
	outIdx := strings.Index(got, "## [EXPECTED OUTPUT]")
	if outIdx < 0 {
		t.Fatalf("missing [EXPECTED OUTPUT] block")
	}
	tail := got[outIdx:]
	for _, want := range []string{"input_unparseable", "exactly once", "NEVER score numerically"} {
		if !strings.Contains(tail, want) {
			t.Errorf("[EXPECTED OUTPUT] missing %q", want)
		}
	}
}

// -----------------------------------------------------------------------------
// ADR-197 override routing is uniform across modes
// -----------------------------------------------------------------------------

func TestComposeCritic_setMode_taskOverridableContractNot(t *testing.T) {
	ctx := criticSetGoldenCtx()
	ctx.Overrides = map[string]string{"task": "OPERATOR TASK OVERRIDE"}
	got := ComposeCriticInstruction(CriticStep(), ctx)

	if !strings.Contains(got, "OPERATOR TASK OVERRIDE") {
		t.Fatalf("set-mode task must route through the ADR-197 override merge")
	}
	if strings.Contains(got, "H1. **Option count**") {
		t.Fatalf("overridden task must replace the embedded rubrics")
	}
	// The wire contract survives any task override.
	outIdx := strings.Index(got, "## [EXPECTED OUTPUT]")
	if outIdx < 0 || !strings.Contains(got[outIdx:], `{"verdicts":`) {
		t.Fatalf("[EXPECTED OUTPUT] verdicts contract must survive a task override")
	}
	if !strings.Contains(got[outIdx:], "exactly once") {
		t.Fatalf("echo contract must survive a task override")
	}
}

// -----------------------------------------------------------------------------
// CriticConditions: set-mode discriminants
// -----------------------------------------------------------------------------

func TestCriticConditions_setMode(t *testing.T) {
	c := CriticConditions(criticSetState())
	if c == nil {
		t.Fatalf("conditions must build for set-mode state")
	}
	if c["set_mode"] != "true" {
		t.Fatalf("set-mode conditions must surface set_mode=true, got %v", c)
	}
	if _, ok := c["question_type"]; ok {
		t.Fatalf("set-mode conditions must omit the moot question_type, got %v", c)
	}
}

// -----------------------------------------------------------------------------
// Golden byte-pin of the composed set prompt
// -----------------------------------------------------------------------------

func TestComposeCritic_SetGoldenByteIdentical(t *testing.T) {
	got := ComposeCriticInstruction(CriticStep(), criticSetGoldenCtx())
	path := filepath.Join("testdata", "golden_critic_set.txt")
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
		t.Fatalf("composed set-mode critic prompt drifted from %s - regenerate with -update if intentional", path)
	}
}
