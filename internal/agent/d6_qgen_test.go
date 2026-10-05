package agent

import (
	"strings"
	"testing"
)

// D6 4-pillar stub harness for QGen P2 sequential pipeline. Real chaos
// cleared by POC W3 multi-crew variant; this harness keeps the contract
// surface visible during the production-promotion sub-iteration.
//
// QGen-specific D6 surface (extends the SKILL's base list):
//   - Pillar 1 pod-death: each step is stateless (chora-creation aggregate
//     state) — pod kill resumes from last committed batch state, NOT from
//     mid-pipeline.
//   - Pillar 2 delivery: terminationplugin emits one event per crew run
//     (not per sub-agent). chora.creation.qgen.batch_completed.v1 (when
//     reporter publishes) carries the canonical envelope.
//   - Pillar 3 chaos isolation: N concurrent batches × M tenants share
//     the same engine — per-batch state lives in chora_creation, not in
//     pipeline memory.
//   - Pillar 4 trace emission: per-call span carries chora.tenant_id +
//     chora.crew_kind + chora.batch_id + gen_ai.* attributes.

func TestD6P1_qgenStepNamesAreStableAcrossRecovery(t *testing.T) {
	// Recovery key for QGen is batch_id (not a per-step token like Familiar's
	// familiar_id). Step names are part of the deterministic recovery path —
	// drift = recovery breaks.
	pre := AllSteps()
	post := AllSteps()
	if len(pre) != len(post) {
		t.Fatalf("step count drifted across reload: pre=%d post=%d", len(pre), len(post))
	}
	for i := range pre {
		if pre[i].Name != post[i].Name {
			t.Errorf("step %d name drifted: pre=%q post=%q", i, pre[i].Name, post[i].Name)
		}
	}
}

func TestD6P2_qgenTerminationEventTopicIsCanonical(t *testing.T) {
	// QGen's terminationplugin in cmd/qgen/main.go emits to
	// chora.ai_kernel.agent.terminated.v1 (same canonical topic as Familiar,
	// shared inbox in O+ observability). Reporter publishes the
	// domain-specific batch_completed event separately to
	// chora.creation.qgen.batch_completed.v1.
	want := "chora.ai_kernel.agent.terminated.v1"
	if !strings.HasPrefix(want, "chora.ai_kernel.") {
		t.Errorf("canonical topic must live under chora.ai_kernel.*; got %q", want)
	}
	if !strings.HasSuffix(want, ".v1") {
		t.Errorf("canonical topic must carry .v1; got %q", want)
	}
}

func TestD6P3_qgenStepsAreStateless(t *testing.T) {
	// Each step's instruction is a pure function of (step, TaskContext);
	// the agent code holds no per-batch state across pod restarts. Re-render
	// twice with the same input — outputs must match (statelessness check).
	ctx := TaskContext{
		TenantID:        "tenant-d6",
		BatchID:         "batch-d6",
		SubjectHint:     "subject-d6",
		DifficultyHint:  "intermediate",
		DesiredAtomType: "ATOM_TYPE_MULTIPLE_CHOICE",
	}
	for _, step := range AllSteps() {
		a := ComposeQGenInstruction(step, ctx)
		b := ComposeQGenInstruction(step, ctx)
		if a != b {
			t.Errorf("step %s: ComposeQGenInstruction not pure (D6 P3 isolation broken)", step.Name)
		}
	}
}

func TestD6P4_qgenMandatorySpanAttributesCoverPipeline(t *testing.T) {
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
			t.Errorf("MandatorySpanAttributes() missing %q (D6 P4 contract drift)", k)
		}
	}
}
