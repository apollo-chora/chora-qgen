// Package agent holds the QGen P2 sequential pipeline's per-step CREATE
// instruction composer + step-name registry + D6 attribute contract.
//
// QGen is a 6-step content gate per crew-composition SKILL §3 + §3.1:
//
//  1. Assurance       (T3 Flash Lite)   — pre-generation safety/policy gate
//  2. Fitness         (T2 Flash Preview)— rubric + difficulty + taxonomy validator
//  3. WebResearcher   (T2 Flash Preview)— CONDITIONAL: invoked only when the
//     upstream Fitness verdict flags
//     insufficient internal grounding.
//     Cheap no-op fast path otherwise.
//  4. Generation      (T2 Flash Preview)— produces N candidate Q&A
//  5. Evaluation      (T1 Pro Preview)  — LLM-as-judge scoring + ranking
//  6. Delivery        (T3 Flash Lite)   — packages survivors + publishes
//     chora.creation.qgen.batch_completed.v1
//
// 5 canonical reusable-agent roles per skill §3 + 1 conditional WebResearcher
// agent that fills the canonical "external grounding when classifier
// confidence is low" gap. The conditional skip rule lives in the
// WebResearcher TaskBlock — it returns an empty evidence list with reason
// `high_confidence_classification` when invoked unnecessarily, preventing
// downstream LLM-token spend.
//
// Each sub-agent runs as `llmagent.New(...)` inside a `sequentialagent.New(...)`
// wrapper (per crew-composition SKILL §1 Pattern P2).
//
// Unlike Familiar (which has a session-time per-instance config layer per
// ADR-147 §7), QGen is author-facing and stateless w.r.t. per-user variance —
// the only inputs are TenantID + BatchID + SubjectHint + DifficultyHint +
// DesiredAtomType. There is NO learner_persona axis for QGen.
package agent

import (
	"fmt"
	"strings"
)

// Step is a canonical QGen sub-agent identity. Name is stable across
// recoveries (D6 P1) and matches the chora.qgen.step span attribute (D6 P4).
type Step struct {
	Name        string
	Description string
	// RoleBlock is the role-specific [ROLE] content for the CREATE composer.
	RoleBlock string
	// TaskBlock is the role-specific [TASK] content for the CREATE composer.
	TaskBlock string
	// OutputBlock is the role-specific [EXPECTED OUTPUT] content for the
	// CREATE composer. It declares the schema of the step's output JSON
	// (informally — the runtime contract lives in main.go's sub-agent
	// wiring; full Protobuf wire schema is Iter 2.5 scope).
	OutputBlock string
}

// TaskContext is the per-batch context the CREATE composer weaves into the
// [CONTEXT] block (visible to every sub-agent in the pipeline).
type TaskContext struct {
	TenantID        string
	BatchID         string
	SubjectHint     string
	DifficultyHint  string
	DesiredAtomType string
}

// AllSteps returns the 6 sub-agents in pipeline order (5 canonical reusable
// roles per crew-composition SKILL §3 + 1 conditional WebResearcher per §3.1).
// Order is part of the recovery contract — never reorder without a coordinated
// chora-creation outbox + cmd/qgen/main.go change.
func AllSteps() []Step {
	return []Step{
		StepAssurance(),
		StepFitness(),
		StepWebResearcher(),
		StepGeneration(),
		StepEvaluation(),
		StepDelivery(),
	}
}

// StepAssurance — step 1 (canonical Content Assurance, T3): pre-generation
// safety + policy gate per crew-composition SKILL §3.
func StepAssurance() Step {
	return Step{
		Name:        "qgen_assurance",
		Description: "Pre-generation safety/policy gate. Refuse unsafe or out-of-scope batches BEFORE they consume LLM budget.",
		RoleBlock: "You are the Content Assurance sub-agent of the QGen pipeline (canonical reusable role " +
			"per crew-composition SKILL §3). Your job is to refuse unsafe, out-of-scope, or tenant-policy-" +
			"violating batch requests BEFORE they consume any downstream LLM budget. You are NOT generating " +
			"content; you are gating.",
		TaskBlock: "- Check that the subject_hint is within the tenant's allowed content domains.\n" +
			"- Check that the difficulty_hint is one of {foundation, intermediate, advanced}.\n" +
			"- Check that the desired_atom_type is supported by the current platform release.\n" +
			"- Refuse PII-bearing subject_hints (regex-detect emails, phone numbers, GCIDs).\n" +
			"- NEVER proceed if validity is false — emit reason + halt the pipeline.",
		OutputBlock: "JSON: {\"valid\": bool, \"reason\": string}. " +
			"On valid=true, downstream sub-agents proceed; on valid=false, the pipeline halts.",
	}
}

// StepFitness — step 2 (canonical Content Fitness, T2): rubric + difficulty +
// taxonomy validator producing FitSpec for the generator.
func StepFitness() Step {
	return Step{
		Name:        "qgen_fitness",
		Description: "Validate input fits rubric + difficulty + atom taxonomy. Produce FitSpec constraints + signal whether external research is needed.",
		RoleBlock: "You are the Content Fitness sub-agent of the QGen pipeline (canonical reusable role per " +
			"crew-composition SKILL §3). Your job is to map the free-form subject_hint to the canonical " +
			"Chora taxonomy AND produce a FitSpec the generator consumes AND signal whether the downstream " +
			"conditional WebResearcher must be invoked.",
		TaskBlock: "- Resolve subject_hint to the closest Chora taxonomy node (e.g., \"agile-estimation\" " +
			"→ \"software-engineering > project-management > agile-estimation\").\n" +
			"- Confirm or downgrade the difficulty_hint based on the resolved taxonomy node's typical level.\n" +
			"- Assign a 0.0-1.0 confidence score; below 0.5 flag the batch for human review.\n" +
			"- Set needs_web_research=true ONLY when (a) confidence < 0.85 OR (b) subject_node is not on " +
			"the platform's evergreen list (covered atoms with stable canonical answers). Default false.",
		OutputBlock: "JSON: {\"subject_node\": string (canonical path), \"difficulty\": string " +
			"(foundation|intermediate|advanced), \"confidence\": float in [0.0, 1.0], " +
			"\"needs_web_research\": bool}.",
	}
}

// StepWebResearcher — step 3 (CONDITIONAL, T2 when invoked): external evidence
// retrieval. Per crew-composition SKILL §3.1 conditional-agent pattern, this
// agent runs in the sequential pipeline but emits a cheap no-op when the
// Fitness verdict's needs_web_research=false. The pipeline cost stays close
// to 5 canonical agents in the common case (high-confidence taxonomy).
func StepWebResearcher() Step {
	return Step{
		Name:        "qgen_web_researcher",
		Description: "CONDITIONAL: external web-evidence retrieval. Skipped (returns empty evidence + skip reason) when the upstream Fitness verdict says needs_web_research=false.",
		RoleBlock: "You are the conditional Web Researcher sub-agent of the QGen pipeline (the 6th step, " +
			"non-canonical, per crew-composition SKILL §3.1 conditional-agent pattern). Your job is to " +
			"ground downstream generation in cited, retrievable evidence — but ONLY when the upstream " +
			"Fitness verdict says needs_web_research=true. When the verdict says false, you exit cheaply.",
		TaskBlock: "**CONDITIONAL SKIP RULE — evaluate FIRST**:\n" +
			"If the upstream Fitness verdict has needs_web_research=false, return immediately:\n" +
			"  {\"evidence\": [], \"reason\": \"high_confidence_classification\", \"invoked\": false}\n" +
			"Do NOT use any web tools. Do NOT spend output tokens beyond the no-op return.\n" +
			"\n" +
			"When needs_web_research=true:\n" +
			"- Use the web search tool (when available) to retrieve 3-5 authoritative sources for the " +
			"classified subject_node.\n" +
			"- For each evidence item: capture title, URL, publication date, 1-sentence excerpt.\n" +
			"- Reject sources older than 5 years for technology topics; older than 10 years for foundational.\n" +
			"- NEVER fabricate a URL — if no source is retrievable, return an empty evidence list with " +
			"reason=no_authoritative_sources.",
		OutputBlock: "JSON: {\"evidence\": [{\"title\": string, \"url\": string, \"published_at\": ISO-8601, " +
			"\"excerpt\": string}], \"reason\": string, \"invoked\": bool}. " +
			"Always carries the `invoked` boolean so downstream evidence is auditable per ADR-141 D2.",
	}
}

// StepGeneration — step 4 (canonical Content Generation, T2): produces N
// candidate questions per FitSpec + any conditional web evidence.
func StepGeneration() Step {
	return Step{
		Name:        "qgen_generation",
		Description: "Produce N candidate Q&A pairs from FitSpec + (optionally) conditional web evidence.",
		RoleBlock: "You are the Content Generation sub-agent of the QGen pipeline (canonical reusable role " +
			"per crew-composition SKILL §3). Your job is to draft candidate question + answer pairs of " +
			"the desired_atom_type, each grounded in either (a) the conditional Web Researcher's evidence " +
			"when it was invoked, OR (b) the canonical Chora taxonomy node's evergreen content when " +
			"web research was skipped.",
		TaskBlock: "- Produce 5 candidate Q&A pairs matching desired_atom_type (e.g., MULTIPLE_CHOICE: 1 stem + 4 options).\n" +
			"- When web_researcher.invoked=true: each candidate cites at least one evidence_url.\n" +
			"- When web_researcher.invoked=false: each candidate cites the canonical taxonomy node + a " +
			"learner-accessible reference (e.g., the platform's atom_id when one exists; otherwise the " +
			"canonical subject_node path itself).\n" +
			"- Difficulty MUST match the Fitness-resolved difficulty (NOT the original hint).",
		OutputBlock: "JSON: {\"candidates\": [{\"question\": string, \"answer\": string, \"distractors\": " +
			"[string] (MCQ only), \"citations\": [string]}]}. Length 0-5 candidates.",
	}
}

// StepEvaluation — step 5 (canonical Content Evaluation, T1 LLM-as-judge):
// scores candidates against the rubric, rejects below threshold.
func StepEvaluation() Step {
	return Step{
		Name:        "qgen_evaluation",
		Description: "LLM-as-judge: score candidates on factuality + clarity + difficulty-alignment; reject below threshold.",
		RoleBlock: "You are the Content Evaluation sub-agent of the QGen pipeline (canonical reusable role " +
			"per crew-composition SKILL §3). Your job is to LLM-as-judge each candidate against the " +
			"rubric. You are NOT generating content; you are grading.",
		TaskBlock: "- Score each candidate on three axes:\n" +
			"  1. Factuality (does the answer match the citations?)\n" +
			"  2. Clarity (would a learner at the target difficulty understand?)\n" +
			"  3. Difficulty alignment (does the candidate actually match the resolved difficulty?)\n" +
			"- Each axis: 0.0-1.0. Composite score = mean.\n" +
			"- Reject candidates with composite < 0.6 (do NOT just downrank).\n" +
			"- Output the surviving candidates sorted by composite score descending.",
		OutputBlock: "JSON: {\"scored\": [{\"candidate\": <Q&A pair>, \"factuality\": float, \"clarity\": float, " +
			"\"difficulty\": float, \"composite\": float}]} sorted by composite descending. " +
			"Rejected candidates omitted entirely.",
	}
}

// StepDelivery — step 6 (canonical Content Delivery, T3): packages survivors
// + publishes chora.creation.qgen.batch_completed.v1.
func StepDelivery() Step {
	return Step{
		Name:        "qgen_delivery",
		Description: "Package the surviving candidates + publish chora.creation.qgen.batch_completed.v1.",
		RoleBlock: "You are the Content Delivery sub-agent of the QGen pipeline (canonical reusable role " +
			"per crew-composition SKILL §3). Your job is to assemble the final batch report + publish the " +
			"canonical event so downstream chora-creation persists the surviving candidates as " +
			"DRAFT atoms.",
		TaskBlock: "- Take the top-N (default 3) scored candidates from the Evaluation step.\n" +
			"- Assemble a BatchReport with: tenant_id, batch_id, subject_node, candidates, " +
			"per-candidate scores, evidence_urls, model_provenance, web_researcher_invoked (bool).\n" +
			"- Call the publish_qgen_batch_completed tool to emit chora.creation.qgen.batch_completed.v1.\n" +
			"- NEVER fabricate provenance — model + token usage MUST come from the actual sub-agent telemetry.",
		OutputBlock: "JSON: {\"report\": <BatchReport>, \"publish_status\": \"published\"|\"failed\"|\"skipped\", " +
			"\"published_message_id\": string}. The pipeline returns this object to the calling caller.",
	}
}

// ComposeQGenInstruction emits the deterministic 6-block CREATE prompt
// for the given step + task context. Pure function — same input always
// yields the same output (IMDA D2 transparency per ADR-141).
func ComposeQGenInstruction(step Step, ctx TaskContext) string {
	var b strings.Builder

	// [CONTEXT] — platform framing + per-batch task context.
	b.WriteString("## [CONTEXT]\n")
	b.WriteString("You are running inside the Chora QGen content gate — a 6-step author-facing pipeline " +
		"that produces grounded, audited Q&A atoms for the Chora learning platform. Every output is " +
		"audited against IMDA Model AI Governance criteria.\n")
	fmt.Fprintf(&b, "Batch context: tenant_id=%s, batch_id=%s, subject_hint=%s, difficulty_hint=%s, desired_atom_type=%s.\n\n",
		safe(ctx.TenantID, "<unset>"), safe(ctx.BatchID, "<unset>"),
		safe(ctx.SubjectHint, "<unset>"), safe(ctx.DifficultyHint, "<unset>"),
		safe(ctx.DesiredAtomType, "<unset>"))

	// [ROLE] — step-specific identity.
	b.WriteString("## [ROLE]\n")
	b.WriteString(step.RoleBlock)
	b.WriteString("\n\n")

	// [EXAMPLES] — minimal, per-step. Sandbox-grade; M14 expands to fixture
	// matrix per QGen 5-role basic package per crew-composition SKILL §3.
	b.WriteString("## [EXAMPLES]\n")
	b.WriteString("Example output format for this step (Iter 2 sandbox; full fixtures land at Iter 2.5):\n")
	fmt.Fprintf(&b, "  Step name: %s\n", step.Name)
	fmt.Fprintf(&b, "  Step description: %s\n\n", step.Description)

	// [AUDIENCE] — fixed for QGen: downstream pipeline sub-agents read the
	// output, NOT a learner. Different sub-agents have different downstream
	// readers, but the framing is uniform: structured JSON.
	b.WriteString("## [AUDIENCE]\n")
	b.WriteString("Your output is consumed by the next sub-agent in the pipeline (and ultimately by " +
		"chora-creation when the Reporter publishes the batch). NEVER write conversational text — only " +
		"the structured JSON declared in the output schema below.\n\n")

	// [TASK] — step-specific invariants.
	b.WriteString("## [TASK]\n")
	b.WriteString(step.TaskBlock)
	b.WriteString("\n\n")

	// [EXPECTED OUTPUT] — step-specific schema + universal anti-patterns.
	b.WriteString("## [EXPECTED OUTPUT]\n")
	b.WriteString(step.OutputBlock)
	b.WriteString("\n")
	b.WriteString("NEVER write commentary outside the JSON. NEVER fabricate citations, atom_ids, " +
		"or scores. If you cannot complete the step, return an explicit error object: " +
		"{\"error\": \"reason\"}.\n")

	return b.String()
}

// MandatorySpanAttributes is the canonical OTel span attribute list every
// QGen span MUST carry. Per agentic-resilience-d6 SKILL Pillar
// 4 + ADR-141 D1 accountability. Drift = D6 P4 promotion gate failure.
func MandatorySpanAttributes() []string {
	return []string{
		"chora.tenant_id",
		"chora.batch_id",
		"chora.crew_kind",
		"chora.qgen.step",
		"gen_ai.request.model",
		"gen_ai.usage.output_tokens",
	}
}

func safe(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}
