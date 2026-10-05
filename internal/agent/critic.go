// Package agent — qgen_critic prompt composer for the 2-agent qgen crew.
//
// The qgen 2-agent crew is the agentic backing for chora-creation's
// `POST /api/atoms/ai-assist` async surface (per docs/m13/ack-oe-ai-assist-
// plan-2026-05-17.md). It is DISTINCT from:
//
//   - qgen_pipeline      — 6-agent batch content gate (chora.creation.qgen.*)
//   - qgen_question      — 3-agent single-question CR pipeline per ADR-153
//   - ai_assist_crew: the legacy 6-agent gate (validator, classifier,
//     web-researcher, qa-generator, evaluator, reporter). RETIRED: its
//     orchestrators/ai_assist_crew.py was deleted 2026-08-23 with the rest of
//     the dead HTTP surface, superseded by this 2-agent crew.
//
// Why a separate 2-agent crew (user-locked 2026-05-17):
//   - The legacy 6-agent gate is over-engineered for /api/atoms/ai-assist
//     quick suggestions; latency budget is tighter (~30-60s end-to-end).
//   - The "evaluator" label in the 6-agent crew is reserved for scoring +
//     content-gate concerns; this crew's second role plays **critic**, NOT
//     scoring (user clarification: "I'll have a different scoring agent —
//     different concern").
//   - LangGraph orchestrator wires two engines: qgen_question (LIVE,
//     generator) + qgen_critic (NEW, this) with a ≤max_retries quality loop.
//
// Shape:
//
//	generate (qgen_question gRPC)
//	   → critique (qgen_critic gRPC, this agent)
//	   → quality_gate (orchestrator)
//	        ├── accepted=true             → publish completed.v1
//	        ├── attempt < max_retries     → loop to generate w/ critic_notes
//	        └── attempt = max_retries     → publish completed.v1 +
//	                                         quality_warning + critic_notes
//
// The critic agent is **tool-free** and **single-shot** — it reads one
// AiAssistCandidate (per chora-contracts OpenAPI AiAssistCandidate schema) +
// emits one CritiqueOutput. No retries inside the agent; the orchestrator
// owns the loop.
//
// Per ADR-141 D2 transparency the composer is a pure function — same input
// always yields the same prompt string.
package agent

import (
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------------
// Critic task context — per-call inputs
// -----------------------------------------------------------------------------

// CriticTaskContext carries the per-call context for the qgen_critic agent.
// The orchestrator constructs this on each critique invocation (one per
// attempt in the quality loop).
type CriticTaskContext struct {
	// TenantID + AuthorGCID for span attribution + IMDA D1
	// accountability evidence.
	TenantID   string
	AuthorGCID string

	// JobID (UUIDv7) — also serves as the orchestrator's correlation_id.
	JobID string

	// QuestionType drives which critique rubric the critic applies.
	// Phyllis scope: mcq + oe; reserved sentinels in QuestionType enum
	// route through but the critic refuses them with reason=unsupported.
	QuestionType QuestionType

	// AttemptIndex is the 0-based attempt counter (0 = first attempt).
	// Surfaced in [CONTEXT] so the critic can be slightly more lenient on
	// re-attempts that already incorporated prior critic_notes.
	AttemptIndex int

	// MaxAttempts is the quality-loop budget (default 4: 1 initial + 3
	// retries). Surfaced for transparency only; the orchestrator owns the
	// loop decision.
	MaxAttempts int

	// PriorCriticNotes carries the critic_notes from the previous attempt
	// (empty on AttemptIndex=0). The critic uses this to check whether the
	// regenerator addressed prior concerns.
	PriorCriticNotes string

	// AuthorPrompt is the original user prompt (for re-grounding — the
	// critic verifies the candidate honours the author's intent).
	AuthorPrompt string

	// SubjectHint + CognitiveLevelHint + DifficultyHint are pass-through
	// from AiAssistGenerateRequest.metadata; the critic uses these to
	// verify the candidate matches the requested shape.
	SubjectHint        string
	CognitiveLevelHint string
	DifficultyHint     string

	// CandidateJSON is the verbatim candidate question JSON the critic must
	// review (the qgen_question generator's output for this attempt). The
	// orchestrator stamps it into ADK session state under "input_payload";
	// the per-turn InstructionProvider (cmd/qgen_critic) reads it via
	// BuildCriticTaskContextFromState and ComposeCriticInstruction embeds it
	// in the [CANDIDATE TO CRITIQUE] block. The user message is only a
	// "BEGIN" trigger, so WITHOUT this the critic sees no candidate at all —
	// the exact bug fixed 2026-06-01 (the deferred "M14.2" InstructionProvider
	// was never built, so the critic answered "input_unparseable" and forced
	// every job to max_attempts).
	CandidateJSON string

	// SetMode switches the critique from ONE candidate to a SET of candidates
	// in one call (CHO-2397, ADR-251 D2/D3). When false, the default and the
	// LIVE single-candidate path, every block is byte-for-byte unchanged
	// (proven by the goldens). When true, CandidateJSON carries a
	// {"candidates": [...]} array whose elements each hold an orchestrator-
	// assigned candidate_id and self-declare question_type; the output schema
	// becomes the {"verdicts": [...]} wrapper with a strict candidate_id echo
	// contract the orchestrator's parse_critique_set_response enforces
	// fail-loud. The single QuestionType field is moot in set mode (mirrors
	// the generator's SetMode contract in composer_question.go).
	SetMode bool

	// Overrides carries operator-supplied SAFE-block replacements (ADR-197 M-B)
	// keyed by segment_id. Only the SAFE blocks are overridable: "role", "task",
	// "examples". The [EXPECTED OUTPUT] JSON-contract block and the safety
	// preamble are NEVER routed through an override. Absent / nil / empty ⇒ the
	// embedded blocks are used, keeping the composed prompt byte-identical to
	// pre-override behaviour. Read from prompt_overrides_json in session state by
	// BuildCriticTaskContextFromState.
	Overrides map[string]string

	// ResolvedPromptVersion is the orchestrator-resolved prompt version (ADR-197
	// M-B). Empty ⇒ the agentconfig-embedded version is the fallback (current
	// behaviour). Span stamping is owned by promptstamping.WithStamping; this
	// field carries the same value into the composer for completeness.
	ResolvedPromptVersion string
}

// -----------------------------------------------------------------------------
// Critic single-step descriptor
// -----------------------------------------------------------------------------

// CriticStep returns the critic agent's static Step descriptor (no per-
// QuestionType branching at the Step level — the composer handles that in
// the TaskBlock + OutputBlock per QuestionType in ComposeCriticInstruction).
//
// Tier 1 (gemini-3.1-pro-preview) — same tier as qgen_question's
// internal evaluation, but with **critic** (not scoring) semantics.
func CriticStep() Step {
	return Step{
		Name: "qgen_critic",
		Description: "Qualitative critique of one candidate question from the qgen_question " +
			"generator. Single-shot; NOT scoring; emits accept/reject + critique_notes + " +
			"suggested_revisions which the LangGraph orchestrator uses to drive the " +
			"≤max_retries quality loop. Distinct from the 6-agent crew's `evaluator` " +
			"(scoring) per user clarification 2026-05-17 — naming reflects that.",
		RoleBlock: "You are the Critic agent of the 2-agent qgen crew — the second of two " +
			"members (qgen_question generates, you critique). You are NOT scoring (no " +
			"numeric verdict). You are NOT regenerating (the orchestrator owns the loop). " +
			"You are NOT the 6-agent gate's `evaluator` (different concern entirely). Your " +
			"single responsibility: read the candidate the generator just produced, decide " +
			"whether it is good enough to surface to the author, and if not, write " +
			"actionable critique notes the regenerator can act on.",
		// Per-QuestionType TaskBlock is resolved inside the composer — keeping
		// the placeholder uniform with the 3-agent crew's Step shape.
		TaskBlock:   "<resolved per QuestionType by ComposeCriticInstruction>",
		OutputBlock: "<resolved per QuestionType by ComposeCriticInstruction>",
	}
}

// -----------------------------------------------------------------------------
// ComposeCriticInstruction — pure function (D2 transparency)
// -----------------------------------------------------------------------------

// ComposeCriticInstruction emits the deterministic 6-block CREATE prompt for
// the critic agent. The TaskBlock + OutputBlock branch per QuestionType
// (mcq + oe) so the LLM applies the right rubric. Pure function (same input
// always yields the same output) per ADR-141 D2 transparency.
//
// The candidate itself is NOT embedded in the composed prompt — the
// LangGraph orchestrator passes it at runtime as the agent's input message
// (per ADK Go llmagent contract). The composed instruction tells the LLM
// HOW to parse + critique the incoming candidate JSON.
func ComposeCriticInstruction(step Step, ctx CriticTaskContext) string {
	task, output := step.TaskBlock, step.OutputBlock
	if step.Name == "qgen_critic" {
		task, output = resolveCriticTaskAndOutput(ctx)
	}

	var b strings.Builder

	// [CONTEXT] — platform framing + per-call task context.
	b.WriteString("## [CONTEXT]\n")
	b.WriteString("You are running inside the Chora qgen 2-agent crew — the AI-assist quality-loop " +
		"pipeline for /api/atoms/ai-assist (chora-creation). This crew is DISTINCT from the 6-agent " +
		"ai_assist_crew (legacy content gate) and from qgen_question (3-agent generator pipeline). " +
		"Every output is audited against IMDA Model AI Governance criteria — D1 accountability + " +
		"D2 transparency in particular.\n")
	fmt.Fprintf(&b, "Call context: tenant_id=%s, author_gcid=%s, job_id=%s, question_type=%s, "+
		"attempt=%d/%d.\n",
		safe(ctx.TenantID, "<unset>"),
		safe(ctx.AuthorGCID, "<unset>"),
		safe(ctx.JobID, "<unset>"),
		safe(string(ctx.QuestionType), "<unset>"),
		ctx.AttemptIndex+1,
		zeroOr(ctx.MaxAttempts, 4),
	)
	if strings.TrimSpace(ctx.AuthorPrompt) != "" {
		fmt.Fprintf(&b, "Author prompt: %s\n", ctx.AuthorPrompt)
	}
	if hints := metadataHints(ctx); hints != "" {
		fmt.Fprintf(&b, "Metadata hints: %s\n", hints)
	}
	if strings.TrimSpace(ctx.PriorCriticNotes) != "" {
		fmt.Fprintf(&b, "Prior critic_notes (from the previous attempt — verify the regenerator addressed these): %s\n",
			ctx.PriorCriticNotes)
	}
	b.WriteString("\n")

	// [ROLE] — step-specific identity. ADR-197 M-B: SAFE block, operator-overridable.
	b.WriteString("## [ROLE]\n")
	b.WriteString(overrideOr(ctx.Overrides, "role", step.RoleBlock))
	b.WriteString("\n\n")

	// [EXAMPLES] — minimal, concrete. No fabricated content. Examples include
	// hard-criteria rejection patterns so the LLM short-circuits objectively
	// before grading subjective quality (tightened 2026-05-17 per the live
	// integration test_*_critique_qualitative_only smokes).
	// ADR-197 M-B: SAFE block, operator-overridable (the embedded body is
	// assembled into exBuf so the no-override path stays byte-identical).
	// Set mode swaps in verdicts-array examples: the single-shape examples
	// carry concrete single-verdict JSON that would contradict the set output
	// contract (unlike the generator, whose examples are shape-neutral).
	b.WriteString("## [EXAMPLES]\n")
	var exBuf strings.Builder
	if ctx.SetMode {
		exBuf.WriteString(criticSetExamples())
	} else {
		writeCriticSingleExamples(&exBuf)
	}
	b.WriteString(overrideOr(ctx.Overrides, "examples", exBuf.String()))

	// [AUDIENCE]: orchestrator (NOT learner-facing). Set mode names the strict
	// set parser + per-verdict routing; the single path is byte-identical.
	b.WriteString("## [AUDIENCE]\n")
	if ctx.SetMode {
		b.WriteString(criticSetAudience())
	} else {
		b.WriteString(criticSingleAudience())
	}

	// [TASK]: per-QuestionType invariants (set mode: both rubrics + the echo
	// contract). ADR-197 M-B: SAFE block, operator-overridable in BOTH modes
	// (mirrors the generator; the wire contract lives in [EXPECTED OUTPUT],
	// which is never overridable, so it survives any task override).
	b.WriteString("## [TASK]\n")
	b.WriteString(overrideOr(ctx.Overrides, "task", task))
	b.WriteString("\n\n")

	// [CANDIDATE(S) TO CRITIQUE]: the verbatim JSON the orchestrator stamped
	// into input_payload: one candidate on the single path, the candidates
	// array in set mode. Embedded HERE (not passed as the user message: the
	// executor sends only a "BEGIN" trigger). The InstructionProvider fails
	// loud upstream (BuildCriticTaskContextFromState) when the payload is
	// missing, so this is non-empty on every real call.
	if ctx.SetMode {
		b.WriteString("## [CANDIDATES TO CRITIQUE]\n")
		b.WriteString("Apply the [TASK] rubrics to EXACTLY these candidates (verbatim JSON " +
			"from the qgen_question generator; each element self-declares its question_type):\n")
	} else {
		b.WriteString("## [CANDIDATE TO CRITIQUE]\n")
		b.WriteString("Apply the [TASK] checks to EXACTLY this candidate (verbatim JSON " +
			"from the qgen_question generator):\n")
	}
	b.WriteString(ctx.CandidateJSON)
	b.WriteString("\n\n")

	// [EXPECTED OUTPUT]: per-shape schema + universal anti-patterns. NEVER
	// overridable in either mode: this is the wire contract the orchestrator
	// parses.
	b.WriteString("## [EXPECTED OUTPUT]\n")
	b.WriteString(output)
	b.WriteString("\n")
	if ctx.SetMode {
		b.WriteString("NEVER write commentary outside the JSON. NEVER fabricate options, model_answers, or " +
			"rubric criteria - you are reviewing, not writing. NEVER score numerically (no 0-1 floats, no " +
			"composite, no rubric grades) - qualitative only. If ONE candidate's JSON is unparseable, emit " +
			"its verdict as accepted=false with critique_notes \"input_unparseable: <reason>\" and keep " +
			"every other verdict intact - NEVER omit a candidate_id over one bad candidate. If the ENTIRE " +
			"input is unparseable as a candidates array, return {\"verdicts\": []} so the orchestrator's " +
			"strict parser fails the chunk loudly - never invent candidate_ids.\n")
	} else {
		b.WriteString("NEVER write commentary outside the JSON. NEVER fabricate options, model_answers, or " +
			"rubric criteria — you are reviewing, not writing. NEVER score numerically (no 0-1 floats, no " +
			"composite, no rubric grades) — qualitative only. If you cannot parse the input candidate, " +
			"return {\"accepted\": false, \"critique_notes\": \"input_unparseable: <reason>\", " +
			"\"suggested_revisions\": []}.\n")
	}

	return b.String()
}

// writeCriticSingleExamples assembles the single-candidate [EXAMPLES] body.
// Extracted verbatim from the inline builder so the set branch can swap the
// body without touching the single-path bytes (goldens prove).
func writeCriticSingleExamples(exBuf *strings.Builder) {
	exBuf.WriteString("Example MCQ acceptance: a clear stem (ANY length) + 4 options + each option has a " +
		"non-empty explainer + distractors plausible → {\"accepted\": true, \"critique_notes\": \"Stem " +
		"is clear and answerable; 4 options with one unambiguous correct key and plausible distractors; " +
		"each explainer states why the option is right or wrong.\", \"suggested_revisions\": []}.\n")
	exBuf.WriteString("Example MCQ HARD-REJECTION (option count): candidate has only 2 options → " +
		"{\"accepted\": false, \"critique_notes\": \"H1: only 2 options provided; MCQs require 3-5 to " +
		"test discrimination\", \"suggested_revisions\": [\"add 2 more plausible distractors covering " +
		"common misconceptions about the topic\"]}.\n")
	exBuf.WriteString("Example MCQ HARD-REJECTION (tautological explainers): every option has explainer " +
		"like \"It is a process.\" → {\"accepted\": false, \"critique_notes\": \"H2: explainers are " +
		"tautological — option_id=a explainer 'It is a process.' restates the option label without " +
		"explaining why\", \"suggested_revisions\": [\"rewrite each explainer to state WHY the option is " +
		"correct/incorrect, citing the underlying concept\"]}.\n")
	exBuf.WriteString("Example OE acceptance: model_answer ≥40 words + 3+ rubric criteria + weights sum to " +
		"1.0 ± 1% (or 100 ± 1%) + grader_tier ∈ {T1,T2} → {\"accepted\": true, \"critique_notes\": " +
		"\"Model answer is substantive and accurate; rubric has 3+ well-weighted criteria summing to 1.0; " +
		"grader_tier set for LLM grading.\", \"suggested_revisions\": []}.\n")
	exBuf.WriteString("Example OE HARD-REJECTION (multiple gates): model_answer is 12 words + rubric length " +
		"1 + grader_tier missing → {\"accepted\": false, \"critique_notes\": \"H1: model_answer has 12 " +
		"words; require ≥40. H2: rubric has 1 criterion; require ≥3. H4: grader_tier missing; require " +
		"T1 or T2\", \"suggested_revisions\": [\"expand model_answer to ≥40 words covering the core " +
		"process + 2 supporting factors\", \"add 2 more rubric criteria covering scientific accuracy + " +
		"explanation clarity\", \"set grader_tier=T2 (LLM-graded with rubric)\"]}.\n\n")
}

// criticSingleAudience is the single-candidate [AUDIENCE] body, byte-identical
// to the pre-set-mode inline text (goldens prove).
func criticSingleAudience() string {
	return "Your output is consumed by the chora-ai-kernel-orchestrator's quality_gate node. The " +
		"orchestrator routes on `accepted`: true → publish ai_assist.completed.v1; false + retries left → " +
		"loop to regenerate with your critique_notes as additional context; false + retries exhausted → " +
		"publish completed.v1 with quality_warning + your last critique_notes. NEVER address the author " +
		"directly. On ACCEPT (accepted=true), critique_notes is a concise one-sentence rationale of WHY " +
		"the candidate passed (the key strengths you verified) — this is the IMDA D2 transparency record " +
		"surfaced to auditors in O+ Decision-Traces + Cloud Trace. On REJECT (accepted=false), " +
		"critique_notes is actionable feedback for the next regenerator pass.\n\n"
}

// criticSetAudience is the set-mode [AUDIENCE] body: names the strict set
// parser and the per-verdict routing so the LLM understands one malformed
// echo wastes the whole call.
func criticSetAudience() string {
	return "Your output is consumed by the chora-ai-kernel-orchestrator's parse_critique_set_response " +
		"and quality_gate nodes. The parser enforces the echo contract STRICTLY: every input candidate_id " +
		"echoed exactly once, no extras, no omissions, no duplicates. ANY violation fails the whole chunk " +
		"loudly and wastes the entire call. The quality gate then routes EACH candidate on its own " +
		"verdict: accepted=true publishes it; accepted=false regenerates that candidate with your " +
		"critique_notes carried. NEVER address the author directly. On ACCEPT, critique_notes is a " +
		"concise one-sentence rationale of WHY the candidate passed (the key strengths you verified) - " +
		"the IMDA D2 transparency record surfaced to auditors in O+ Decision-Traces + Cloud Trace. On " +
		"REJECT, critique_notes is actionable feedback for the next regenerator pass.\n\n"
}

// criticSetExamples is the set-mode [EXAMPLES] body: one worked 2-candidate
// mixed example showing the verdicts wrapper and the echo discipline.
func criticSetExamples() string {
	return "Example set critique (2 candidates, one accept + one reject):\n" +
		"  Input: {\"candidates\": [{\"candidate_id\": \"c0\", \"question_type\": \"mcq\", ...a clear " +
		"stem, 4 options, each with a non-empty explainer, one unambiguous correct key...}, " +
		"{\"candidate_id\": \"c1\", \"question_type\": \"oe\", ...model_answer of 12 words, rubric of 1 " +
		"criterion, grader_tier missing...}]}\n" +
		"  Output: {\"verdicts\": [{\"candidate_id\": \"c0\", \"accepted\": true, \"critique_notes\": " +
		"\"Stem is clear and answerable; 4 options with one unambiguous correct key and plausible " +
		"distractors; each explainer states why the option is right or wrong.\", \"suggested_revisions\": " +
		"[]}, {\"candidate_id\": \"c1\", \"accepted\": false, \"critique_notes\": \"H1: model_answer has " +
		"12 words; require >=40. H2: rubric has 1 criterion; require >=3. H4: grader_tier missing; " +
		"require T1 or T2\", \"suggested_revisions\": [\"expand model_answer to >=40 words covering the " +
		"core process + 2 supporting factors\", \"add 2 more rubric criteria covering scientific accuracy " +
		"+ explanation clarity\", \"set grader_tier=T2 (LLM-graded with rubric)\"]}]}\n" +
		"Note the echo discipline: exactly the ids c0 and c1, each exactly once, in any order.\n\n"
}

// -----------------------------------------------------------------------------
// Per-QuestionType template resolution
// -----------------------------------------------------------------------------

// resolveCriticTaskAndOutput selects the per-QuestionType critique template.
// Returns (TaskBlock, OutputBlock). The Phyllis scope is mcq + oe; reserved
// sentinels short-circuit to a refuse-template so the regenerator's contract
// stays unambiguous.
func resolveCriticTaskAndOutput(ctx CriticTaskContext) (task, output string) {
	// Set mode (CHO-2397, ADR-251 D2/D3): critique a whole candidates array in
	// one call. Returns BEFORE the single-candidate switch so that path is
	// untouched (mirrors resolveTaskAndOutput's SetMode branch in
	// composer_question.go).
	if ctx.SetMode {
		return taskCritiqueSet(), outputCritiqueSet()
	}

	switch ctx.QuestionType {
	case QuestionTypeMCQ:
		return taskCritiqueMCQ(), outputCritiqueMCQ()
	case QuestionTypeOE:
		return taskCritiqueOE(), outputCritiqueOE()
	}
	// Unknown question_type — fail loud (no fake template fallback).
	return "// invalid question_type — refuse critique",
		"{\"accepted\": false, \"critique_notes\": \"unsupported_question_type\", \"suggested_revisions\": []}"
}

// taskCritiqueSet is the set-mode [TASK]: the candidates-array framing, the
// candidate_id echo contract, and BOTH per-type rubrics embedded verbatim
// (single-sourced from taskCritiqueMCQ / taskCritiqueOE so the set rubrics can
// never drift from the single-candidate ones).
func taskCritiqueSet() string {
	return "The input is a SET of candidate questions from the qgen_question generator, a JSON object " +
		"shaped:\n" +
		"  {\"candidates\": [{\"candidate_id\": str, \"question_type\": \"mcq\"|\"oe\", ...candidate " +
		"fields...}, ...]}.\n" +
		"Each element carries an orchestrator-assigned `candidate_id` and self-declares its " +
		"`question_type`. Critique EVERY candidate INDEPENDENTLY: one verdict per candidate, and never " +
		"let one candidate's quality colour another's verdict.\n" +
		"\n**ECHO CONTRACT (hard requirement):** your verdicts array MUST contain EXACTLY ONE verdict " +
		"per input candidate_id: every candidate_id sent, echoed back verbatim, exactly once. No ids " +
		"invented, no ids omitted, no duplicates. The orchestrator's parser fails the whole chunk " +
		"loudly on any violation, wasting the entire call.\n" +
		"\nApply the per-type rubric below to each candidate according to its question_type. The " +
		"rubrics are written addressing one candidate (\"the input is one ... candidate\"): read them " +
		"as addressing the candidate currently under review.\n" +
		"\n### MCQ rubric (apply to each candidate with question_type=mcq)\n" +
		taskCritiqueMCQ() +
		"\n\n### OE rubric (apply to each candidate with question_type=oe)\n" +
		taskCritiqueOE()
}

// outputCritiqueSet is the set-mode [EXPECTED OUTPUT] schema: the verdicts
// wrapper with the echo contract restated. This block is never overridable,
// so the wire contract survives any ADR-197 task override.
func outputCritiqueSet() string {
	return "JSON: {\"verdicts\": [{\"candidate_id\": str, \"accepted\": bool, \"critique_notes\": " +
		"string, \"suggested_revisions\": [string, ...]}, ...]}.\n" +
		"  - EXACTLY ONE verdict per input candidate_id (echo contract): every candidate_id from the " +
		"input echoed verbatim, exactly once, no extras, no omissions, no duplicates.\n" +
		"  - accepted=true: critique_notes MUST be a concise 1-sentence rationale naming the key " +
		"strengths you verified (e.g. clear stem, plausible distractors, correct key) - the IMDA D2 " +
		"audit record an auditor reads, NOT learner-facing; suggested_revisions MAY be empty.\n" +
		"  - accepted=false: critique_notes MUST be non-empty (1-3 sentences) citing the specific gap " +
		"(hard-criterion ID or option_id + issue); suggested_revisions SHOULD include at least 1 " +
		"actionable item.\n" +
		"  - A candidate whose own JSON is unparseable gets {\"candidate_id\": <its id>, \"accepted\": " +
		"false, \"critique_notes\": \"input_unparseable: <reason>\", \"suggested_revisions\": []}; " +
		"every other candidate's verdict stays intact."
}

// taskCritiqueMCQ — critique rubric for MCQ candidates.
//
// Tightened 2026-05-17 to reject deliberately-weak MCQs that the prior prompt
// accepted (see integration test test_mcq_critique_qualitative_only). The
// HARD REJECTION CRITERIA block precedes the qualitative 5-check rubric so
// the LLM short-circuits on objective gates before grading subjective quality.
// Candidates may carry options at the top level (`options`) OR nested under
// `mcq_payload.options` — both shapes flow from qgen_question's templates and
// both are valid input here.
func taskCritiqueMCQ() string {
	return "The input is one MCQ candidate from the qgen_question generator. Parse it as JSON shaped per " +
		"AiAssistCandidate in chora-contracts/openapi/creation-questions.yaml:\n" +
		"  {\"stem\": str, \"question_type\": \"mcq\", \"options\": [...], \"mcq_payload\": {\"options\": " +
		"[{\"option_id\": str, \"label\": str, \"text\": str, \"is_correct\": bool, \"explainer\": str}, " +
		"...], \"scoring_mode\": \"single_correct\"|\"multi_correct\"}}.\n" +
		"Options may appear at the top level (`options`) OR nested under `mcq_payload.options` — both are " +
		"valid shapes; apply the checks to whichever list is present.\n" +
		"\n**HARD REJECTION CRITERIA (apply FIRST; any violation → accepted=false):**\n" +
		"  H1. **Option count** — MCQ candidates MUST have **3-5 options**. Fewer than 3 options " +
		"(e.g., only 2 options) is an automatic REJECT — single-distractor MCQs do not test " +
		"discrimination. Cite the actual count in critique_notes (e.g., \"only 2 options provided; MCQs " +
		"require 3-5\").\n" +
		"  H2. **Per-option explainer** — EVERY option MUST carry a non-empty `explainer` field. Missing " +
		"or empty explainers are a REJECT. Tautological explainers (e.g., \"It is a process.\" for " +
		"\"What is a process?\") count as empty and are a REJECT — cite the specific option_id.\n" +
		// NOTE: there is intentionally NO minimum stem word-count gate (user
		// directive 2026-06-01). A concise, clear question ("What gas do plants
		// absorb during photosynthesis?") is a valid MCQ stem — stem QUALITY is
		// judged qualitatively (clarity, check #1 below), never by length.
		"  H3. **Obvious-correct-answer pattern** — if only ONE option is plausibly true at a glance " +
		"(distractors are obviously wrong / clearly off-topic / labels like \"Something else\" / labels " +
		"that are non-answers), it is a REJECT. The whole point of an MCQ is discrimination; an obvious " +
		"correct answer fails the construct.\n" +
		"  H4. **Required option fields** — every option MUST carry `option_id`, `label` (or `text`), " +
		"`is_correct`, and `explainer`. Missing required fields are a REJECT.\n" +
		"\n**Qualitative checks (apply SECOND, only after H1-H4 pass):**\n" +
		"  1. **Stem clarity** — is the stem unambiguous to a learner at the requested cognitive level? " +
		"Judge clarity, NOT length — a short, well-formed question is fine. " +
		"A grammatically-vague stem is a reject. A stem that depends on context not in the prompt is a " +
		"reject.\n" +
		"  2. **Correct answer correctness** — is the marked-correct option actually correct? A factual " +
		"error here is the most serious reject.\n" +
		"  3. **Distractor plausibility** — are the non-correct options plausible enough to test " +
		"discrimination? Distractors that no informed learner would pick are a reject (too easy). " +
		"Distractors that are actually true are a reject (poorly designed).\n" +
		"  4. **Explainer quality** — does each option's explainer give a learner enough to understand " +
		"WHY the answer is right/wrong? Empty explainers are a reject; tautological explainers are a " +
		"reject.\n" +
		"  5. **Single/multi-correct integrity** — for scoring_mode=single_correct, exactly one option " +
		"must be is_correct=true. For multi_correct, ≥1 option must be is_correct=true. Violations are " +
		"rejects.\n" +
		"\nOn rejection, critique_notes MUST cite the SPECIFIC gap (the hard-criterion ID e.g. \"H1: only " +
		"2 options provided; MCQs require 3-5\" OR an option_id + qualitative issue), and " +
		"suggested_revisions MUST give 1-3 actionable directives (e.g., \"add 2 more plausible " +
		"distractors covering common misconceptions\", \"option_id=b: revise distractor — currently a " +
		"true statement which makes it a second correct answer\"). The orchestrator's regenerator reads " +
		"these notes verbatim — vague critique_notes waste a retry slot."
}

// outputCritiqueMCQ — MCQ critique JSON schema.
func outputCritiqueMCQ() string {
	return "JSON: {\"accepted\": bool, \"critique_notes\": string, \"suggested_revisions\": [string, ...]}.\n" +
		"  - accepted=true: critique_notes MUST be a concise 1-sentence rationale naming the key " +
		"strengths you verified (e.g. clear stem, plausible distractors, correct key) — this is the IMDA " +
		"D2 audit record an auditor reads, NOT learner-facing; suggested_revisions MAY be empty.\n" +
		"  - accepted=false: critique_notes MUST be non-empty (1-3 sentences); suggested_revisions SHOULD " +
		"include at least 1 actionable item."
}

// taskCritiqueOE — critique rubric for OE (open-ended) candidates.
//
// Tightened 2026-05-17 to reject deliberately-weak OE candidates that the prior
// prompt accepted (see integration test test_oe_critique_qualitative_only). The
// HARD REJECTION CRITERIA block precedes the qualitative 5-check rubric so the
// LLM short-circuits on objective gates (model_answer word count, rubric size,
// weight-sum tolerance, grader_tier presence) before grading subjective
// quality. Weight sum accepts either fractional (~1.0 ± 1%) per the OpenAPI
// RubricCriterion schema OR percentage (~100 ± 1%) — both shapes are emitted
// in practice and both are valid input here.
func taskCritiqueOE() string {
	return "The input is one OE candidate from the qgen_question generator. Parse it as JSON shaped per " +
		"AiAssistCandidate in chora-contracts/openapi/creation-questions.yaml:\n" +
		"  {\"stem\": str, \"question_type\": \"oe\", \"oe_payload\": {\"model_answer\": str, \"rubric\": " +
		"[{\"criterion_id\": str, \"title\": str, \"description\": str, \"weight\": number}, ...], " +
		"\"grader_tier\": \"T1\"|\"T2\", \"min_response_chars\": int, \"max_response_chars\": int}}.\n" +
		"The candidate's OE fields may appear nested under `oe_payload` OR flat at the top level — both " +
		"shapes flow from qgen_question; apply the checks to whichever shape carries the content.\n" +
		"\n**HARD REJECTION CRITERIA (apply FIRST; any violation → accepted=false):**\n" +
		"  H1. **Model answer word count** — `model_answer` MUST be **at least 40 words** of substantive " +
		"text. A model_answer under 40 words cannot ground T1/T2 grading. Count whitespace-separated " +
		"tokens; cite the actual count in critique_notes (e.g., \"H1: model_answer has 12 words; OE " +
		"model answers require ≥40\").\n" +
		"  H2. **Rubric criterion count** — `rubric` MUST be a list of **at least 3 criteria**. Single-" +
		"criterion or 2-criterion rubrics fail to triangulate partial credit and are a REJECT. Cite the " +
		"actual count.\n" +
		"  H3. **Rubric weights sum tolerance** — `sum(weight)` over all criteria MUST be either " +
		"**1.0 ± 1% (fractional weights)** OR **100 ± 1% (percentage weights)**. Off-by-more (e.g., " +
		"sum=0.5 on a single criterion, or sum=103) is a REJECT — request exact correction citing the " +
		"actual sum and the chosen convention.\n" +
		"  H4. **grader_tier presence + enum** — `grader_tier` MUST be present and MUST be exactly " +
		"\"T1\" or \"T2\". A missing grader_tier is a REJECT (the orchestrator cannot route batched " +
		"grading without it); any other value is a REJECT.\n" +
		"  H5. **Required rubric fields** — every rubric entry MUST carry `criterion_id`, `title`, " +
		"`description`, and `weight`. Missing required fields are a REJECT.\n" +
		"\n**Qualitative checks (apply SECOND, only after H1-H5 pass):**\n" +
		"  1. **Stem clarity** — is the stem unambiguous AND does it solicit a definite answer (not a " +
		"trivia question; not a personal-opinion question if the grader_tier is T1)? Reject ambiguity.\n" +
		"  2. **Model answer accuracy** — is the model_answer factually correct, complete enough for the " +
		"grader_tier, and grounded in the subject the prompt requested? Reject if the model_answer " +
		"hallucinates or contradicts itself.\n" +
		"  3. **Rubric coverage** — do the rubric criteria cover the model_answer's key points? A rubric " +
		"that scores something the model_answer doesn't address (or vice versa) is a reject.\n" +
		"  4. **Rubric weight balance** — beyond the sum-tolerance gate above, are any individual " +
		"weights so dominant they collapse the rubric into a single-criterion grader (e.g., one " +
		"criterion = 0.85)? Reject if so.\n" +
		"  5. **Response-length window plausibility** — `min_response_chars` / `max_response_chars` (if " +
		"present) must realistically bracket what the model_answer length implies for the grader_tier. " +
		"min ≥ max → reject. Wildly asymmetric (e.g., max=10000 for a 50-word answer) → reject.\n" +
		"\nOn rejection, critique_notes MUST cite the SPECIFIC gap (the hard-criterion ID + the actual " +
		"observed value when measurable, e.g. \"H1: model_answer has 12 words; require ≥40\" OR \"H2: " +
		"rubric has 1 criterion; require ≥3\" OR \"H4: grader_tier is missing; require T1 or T2\"). " +
		"suggested_revisions MUST give 1-3 actionable directives the regenerator can apply verbatim " +
		"(e.g., \"expand model_answer to ≥40 words covering chlorophyll's role + at least 2 rate factors\", " +
		"\"add 2 more rubric criteria covering scientific accuracy + clarity; redistribute weights so " +
		"they sum to 1.0\", \"set grader_tier=T2 (LLM-graded with rubric)\"). Vague critique_notes waste " +
		"a retry slot."
}

// outputCritiqueOE — OE critique JSON schema (same shape as MCQ).
func outputCritiqueOE() string {
	return "JSON: {\"accepted\": bool, \"critique_notes\": string, \"suggested_revisions\": [string, ...]}.\n" +
		"  - accepted=true: critique_notes MUST be a concise 1-sentence rationale naming the key " +
		"strengths you verified (e.g. clear stem, plausible distractors, correct key) — this is the IMDA " +
		"D2 audit record an auditor reads, NOT learner-facing; suggested_revisions MAY be empty.\n" +
		"  - accepted=false: critique_notes MUST be non-empty (1-3 sentences); suggested_revisions SHOULD " +
		"include at least 1 actionable item."
}

// -----------------------------------------------------------------------------
// Session-state → CriticTaskContext (runtime per-turn injection, M14.2)
// -----------------------------------------------------------------------------

// BuildCriticTaskContextFromState reads per-turn ADK session state (set by the
// Python executor's _build_session_state for ROLE_QGEN_CRITIC) and assembles
// the CriticTaskContext the composer folds into the prompt — most importantly
// the candidate JSON the critic must review.
//
// State keys read:
//
//	tenant_id          — string (manaplugin passes through)
//	author_gcid        — string (or user_gcid fallback)
//	job_id             — string
//	question_type      — string ("mcq"/"oe"; invalid → fail loud)
//	attempt_index      — int    (0-based; the JSON path hands us float64)
//	max_attempts       — int    (composer defaults to 4 when 0)
//	prior_critic_notes — string (verify the regenerator addressed prior gaps)
//	author_prompt      — string (re-ground the candidate to author intent)
//	subject_hint / cognitive_level_hint / difficulty_hint — string (optional)
//	input_payload      — string (the candidate JSON — REQUIRED; fail loud if empty)
//
// Fail-loud posture (user directive 2026-06-01 — arch-clean / debt-free): a nil
// state, an invalid question_type, or a missing candidate returns an error so
// the InstructionProvider aborts the agent invocation rather than silently
// critiquing nothing. This is the exact bug being closed — the prior boot-time
// static compose never injected the candidate, so the critic answered
// "input_unparseable" and forced the orchestrator quality loop to max_attempts
// on every job.
func BuildCriticTaskContextFromState(state stateGetter) (CriticTaskContext, error) {
	if state == nil {
		return CriticTaskContext{}, fmt.Errorf("qgen_critic: state must not be nil")
	}

	candidate := readStateString(state, "input_payload")
	if strings.TrimSpace(candidate) == "" {
		return CriticTaskContext{}, fmt.Errorf(
			"qgen_critic: no candidate in session state (input_payload empty) — " +
				"the orchestrator must stamp the candidate JSON before critique")
	}

	qtRaw := readStateString(state, "question_type")
	qt := QuestionTypeMCQ
	if qtRaw != "" {
		qt = QuestionType(qtRaw)
	}
	// Set mode (CHO-2397): candidates self-declare question_type, so the
	// single field is moot (the orchestrator may stamp "mixed" or omit it).
	// Skip its validation ONLY then, mirroring the generator's set-mode
	// contract in BuildTaskContextFromState.
	setMode := readStateBool(state, "set_mode")
	if !setMode {
		if err := assertValidQuestionType(qt); err != nil {
			return CriticTaskContext{}, err
		}
	}

	authorGCID := readStateString(state, "author_gcid")
	if authorGCID == "" {
		authorGCID = readStateString(state, "user_gcid")
	}

	return CriticTaskContext{
		TenantID:           readStateString(state, "tenant_id"),
		AuthorGCID:         authorGCID,
		JobID:              readStateString(state, "job_id"),
		QuestionType:       qt,
		AttemptIndex:       readStateInt(state, "attempt_index"),
		MaxAttempts:        readStateInt(state, "max_attempts"),
		PriorCriticNotes:   readStateString(state, "prior_critic_notes"),
		AuthorPrompt:       readStateString(state, "author_prompt"),
		SubjectHint:        readStateString(state, "subject_hint"),
		CognitiveLevelHint: readStateString(state, "cognitive_level_hint"),
		DifficultyHint:     readStateString(state, "difficulty_hint"),
		CandidateJSON:      candidate,
		// Set mode (CHO-2397): false on the dominant single-candidate path.
		SetMode: setMode,
		// ADR-197 M-B (read side) — operator SAFE-block overrides + resolved
		// prompt version. Both absent on the dominant path ⇒ nil / "" ⇒
		// byte-identical to pre-override behaviour.
		Overrides:             readPromptOverridesFromState(state, "prompt_overrides_json"),
		ResolvedPromptVersion: readStateString(state, "resolved_prompt_version"),
	}, nil
}

// readStateInt reads an integer-valued state key, tolerating the float64 the
// executor's JSON path hands us (mirrors asFloat in composer_question.go).
// Returns 0 on miss.
func readStateInt(state stateGetter, key string) int {
	v, err := state.Get(key)
	if err != nil {
		return 0
	}
	return int(asFloat(v))
}

// -----------------------------------------------------------------------------
// Small helpers (kept private to the package; mirror composer.go's safe())
// -----------------------------------------------------------------------------

// metadataHints joins the optional metadata fields into a single human-readable
// line for the [CONTEXT] block. Empty result when no hints are set.
func metadataHints(ctx CriticTaskContext) string {
	var parts []string
	if v := strings.TrimSpace(ctx.SubjectHint); v != "" {
		parts = append(parts, fmt.Sprintf("subject=%s", v))
	}
	if v := strings.TrimSpace(ctx.CognitiveLevelHint); v != "" {
		parts = append(parts, fmt.Sprintf("cognitive_level=%s", v))
	}
	if v := strings.TrimSpace(ctx.DifficultyHint); v != "" {
		parts = append(parts, fmt.Sprintf("difficulty=%s", v))
	}
	return strings.Join(parts, ", ")
}

// zeroOr returns fallback when v == 0; else v.
func zeroOr(v, fallback int) int {
	if v == 0 {
		return fallback
	}
	return v
}
