package boot

// compose.go: qgen_generate mode=compose, the testset composer ported from the
// kennel's orchestrators/testset_composer.py (ADR-254 D2/D6: "testset compose
// rides qgen_generate mode=compose"). Pure and IO-free: the instruction text,
// the source/rubric split, the tolerant JSON parse, the contract-valid
// normalisation and the deterministic fallback title/description, all
// testable without a model.
//
// Wire contract (binding with WP-K, 2026-08-22 18:23Z): input_payload carries
// mode="compose", candidates (JSON array of the accepted candidate payload
// objects, each with its draft_id), author_prompt (the batch prompt), metadata
// (the batch metadata object), source_files (JSON array of {gs_uri, mime_type,
// role}; absent when ungrounded; role "rubric" = the mark scheme, first wins,
// extra rubrics dropped) and grounding_mode. The completion output_payload is
// {"proposed_test_set": {title, description, order, points}} exactly as the
// kennel's _normalise_proposal produced it; an answer that carries no JSON
// object is a permanent FAILED compose_unusable_answer (never a fabricated
// set: the kennel applies its deterministic fallback_proposal on FAILED).

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/apollo-chora/chora-adk-common/agentdispatch"
)

// Contract clamps per chora-contracts/openapi/creation-questions.yaml
// ProposedTestSet (the same constants the kennel composer used).
const (
	ComposeDefaultPoints       = 10
	ComposeTitleMaxChars       = 256
	ComposeDescriptionMaxChars = 2048
	ComposePointsMin           = 1
	ComposePointsMax           = 100
	composeRoleRubric          = "rubric"
	composeStemMaxChars        = 512
	composePromptMaxChars      = 1024
)

// Compose payload keys.
const (
	stateKeyCandidates    = "candidates"
	stateKeyAuthorPrompt  = "author_prompt"
	stateKeySourceFiles   = "source_files"
	stateKeyGroundingMode = "grounding_mode"
	stateKeyComposeRaw    = "qgen_question_compose_raw"
)

// ComposeSystemPrompt is the composer's system prompt (verbatim from the
// kennel, the dash spelled out).
const ComposeSystemPrompt = "You are an assessment designer composing ONE test set from a batch of " +
	"AI-generated questions and the uploaded source material. Respond with " +
	"a single JSON object only, no prose."

// ComposeSourceFile is one grounding file reference.
type ComposeSourceFile struct {
	GSURI    string `json:"gs_uri"`
	MimeType string `json:"mime_type"`
	Role     string `json:"role"`
}

// ComposeRequest is the parsed compose payload.
type ComposeRequest struct {
	Candidates   []map[string]any
	AuthorPrompt string
	SourceFiles  []ComposeSourceFile
	GroundingMod string
}

// anyList reads a payload value that may be a JSON array already decoded
// ([]any) or still a JSON string.
func anyList(v any) ([]any, error) {
	switch t := v.(type) {
	case nil:
		return nil, nil
	case []any:
		return t, nil
	case string:
		if strings.TrimSpace(t) == "" {
			return nil, nil
		}
		var out []any
		if err := json.Unmarshal([]byte(t), &out); err != nil {
			return nil, err
		}
		return out, nil
	}
	return nil, fmt.Errorf("not a list: %T", v)
}

// ParseComposeRequest reads and validates the compose payload; a malformed
// candidates or source_files list is permanent.
func ParseComposeRequest(st stateReader) (ComposeRequest, error) {
	var req ComposeRequest
	rawCands, err := st.Get(stateKeyCandidates)
	if err != nil {
		rawCands = nil
	}
	cands, err := anyList(rawCands)
	if err != nil {
		return req, agentdispatch.Permanent("invalid_candidates: candidates must be a JSON array of candidate objects", err)
	}
	for _, c := range cands {
		if m, ok := c.(map[string]any); ok {
			req.Candidates = append(req.Candidates, m)
		}
	}
	if len(req.Candidates) == 0 {
		return req, agentdispatch.Permanent("missing_candidates: a compose dispatch carries the accepted candidates (each with its draft_id)", nil)
	}
	rawFiles, err := st.Get(stateKeySourceFiles)
	if err != nil {
		rawFiles = nil
	}
	files, err := anyList(rawFiles)
	if err != nil {
		return req, agentdispatch.Permanent("invalid_source_files: source_files must be a JSON array of {gs_uri, mime_type, role}", err)
	}
	for _, f := range files {
		m, ok := f.(map[string]any)
		if !ok {
			continue
		}
		sf := ComposeSourceFile{GSURI: strings.TrimSpace(str(m["gs_uri"])), MimeType: strings.TrimSpace(str(m["mime_type"])), Role: strings.TrimSpace(str(m["role"]))}
		if sf.GSURI == "" {
			continue
		}
		if !strings.HasPrefix(sf.GSURI, "gs://") {
			return req, agentdispatch.Permanent("invalid_source_files: gs_uri must be a gs:// reference", fmt.Errorf("gs_uri=%q", sf.GSURI))
		}
		req.SourceFiles = append(req.SourceFiles, sf)
	}
	req.AuthorPrompt = StateString(st, stateKeyAuthorPrompt)
	req.GroundingMod = StateString(st, stateKeyGroundingMode)
	return req, nil
}

func str(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case nil:
		return ""
	default:
		b, _ := json.Marshal(t)
		return strings.Trim(string(b), `"`)
	}
}

// SplitSourceFiles partitions files into (sources, rubric): the FIRST
// role="rubric" file is the mark scheme, extra rubrics are DROPPED (a mark
// scheme must never masquerade as question material), everything else is a
// source.
func SplitSourceFiles(files []ComposeSourceFile) (sources []ComposeSourceFile, rubric *ComposeSourceFile) {
	for i := range files {
		f := files[i]
		if f.Role == composeRoleRubric {
			if rubric == nil {
				rubric = &f
			}
			continue
		}
		sources = append(sources, f)
	}
	return sources, rubric
}

// DraftIDs lists the candidates' draft ids in submission order (blank dropped).
func DraftIDs(candidates []map[string]any) []string {
	var out []string
	for _, c := range candidates {
		if did := strings.TrimSpace(str(c["draft_id"])); did != "" {
			out = append(out, did)
		}
	}
	return out
}

func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		return string(r[:n])
	}
	return s
}

// ComposeInstructionText is the user-side instruction (the kennel's
// _instruction_text in substance): the candidates as (draft_id, stem,
// question_type), the author's batch request and the exact JSON shape + rules.
func ComposeInstructionText(req ComposeRequest) string {
	type item struct {
		DraftID      string `json:"draft_id"`
		Stem         string `json:"stem"`
		QuestionType string `json:"question_type"`
	}
	items := make([]item, 0, len(req.Candidates))
	for _, c := range req.Candidates {
		items = append(items, item{DraftID: str(c["draft_id"]), Stem: truncateRunes(str(c["stem"]), composeStemMaxChars), QuestionType: str(c["question_type"])})
	}
	itemsJSON, _ := json.Marshal(items)
	return fmt.Sprintf("Compose ONE test set from the generated questions below, mirroring "+
		"the attached source material (title, description, question order) "+
		"and its mark scheme.\n\n"+
		"Author's batch request: %s\n\n"+
		"Generated questions (draft_id, stem, question_type):\n%s\n\n"+
		"Respond with EXACTLY one JSON object:\n"+
		`{"title": "<test set title, max %d chars>", "description": "<1-3 sentence description, max %d chars>", "order": ["<draft_id>", ...], "points": {"<draft_id>": <int %d..%d>}}`+"\n"+
		"Rules:\n"+
		"- order MUST list every draft_id exactly once, in the sequence the source material implies.\n"+
		"- points: extract marks cues from the source / mark scheme (e.g. \"[5 marks]\" or \"Question 3 (10 points)\") and assign each question the marks the source allocates to its counterpart. When the material gives no marks, use %d.\n"+
		"- title/description mirror the source material's own naming where evident.",
		truncateRunes(req.AuthorPrompt, composePromptMaxChars), string(itemsJSON),
		ComposeTitleMaxChars, ComposeDescriptionMaxChars, ComposePointsMin, ComposePointsMax, ComposeDefaultPoints)
}

// ComposeRubricMarker precedes the rubric part: the mark scheme is a DISTINCT
// part behind an explicit marker, never presented as question material.
const ComposeRubricMarker = "The following file is the MARK SCHEME / grading rubric for this paper. " +
	"Use it to extract the per-question marks and align the points distribution to it."

// ComposeUserParts renders the user content: the instruction text, then every
// source as an object-store FileData part, then the rubric marker + rubric (by
// reference, the way companion_extract grounds; the bytes never ride the bus).
func ComposeUserParts(req ComposeRequest) []*genai.Part {
	parts := []*genai.Part{{Text: ComposeInstructionText(req)}}
	sources, rubric := SplitSourceFiles(req.SourceFiles)
	for _, f := range sources {
		parts = append(parts, &genai.Part{FileData: &genai.FileData{FileURI: f.GSURI, MIMEType: mimeOr(f.MimeType)}})
	}
	if rubric != nil {
		parts = append(parts, &genai.Part{Text: ComposeRubricMarker}, &genai.Part{FileData: &genai.FileData{FileURI: rubric.GSURI, MIMEType: mimeOr(rubric.MimeType)}})
	}
	return parts
}

func mimeOr(m string) string {
	m = strings.ToLower(strings.TrimSpace(strings.Split(m, ";")[0]))
	if m == "" {
		return "application/octet-stream"
	}
	return m
}

var composeFenceRE = regexp.MustCompile("(?s)```(?:json)?\\s*(.+?)\\s*```")

// ErrComposeUnusable is the permanent-failure reason token for an answer
// that carries no JSON object.
var ErrComposeUnusable = errors.New("compose_unusable_answer")

// ParseComposeAnswer parses the model completion into an object, tolerating
// ```json fences and leading/trailing prose (first { .. last }).
func ParseComposeAnswer(completion string) (map[string]any, error) {
	text := strings.TrimSpace(completion)
	if text == "" {
		return nil, fmt.Errorf("%w: empty completion", ErrComposeUnusable)
	}
	candidates := []string{text}
	if m := composeFenceRE.FindStringSubmatch(text); m != nil {
		candidates = append(candidates, m[1])
	}
	if first, last := strings.Index(text, "{"), strings.LastIndex(text, "}"); first != -1 && last > first {
		candidates = append(candidates, text[first:last+1])
	}
	for _, c := range candidates {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(c), &parsed); err == nil && parsed != nil {
			return parsed, nil
		}
	}
	return nil, fmt.Errorf("%w: completion carried no JSON object", ErrComposeUnusable)
}

// ProposedTestSet is the contract-valid proposal.
type ProposedTestSet struct {
	Title       string         `json:"title"`
	Description string         `json:"description"`
	Order       []string       `json:"order"`
	Points      map[string]int `json:"points"`
}

// FallbackTitleDescription is the deterministic title/description the
// normalisation uses when the model leaves them blank: the first source's
// file stem, else the author prompt's first 64 chars, else "Generated test
// set"; the description names the candidate count and the grounding file.
func FallbackTitleDescription(req ComposeRequest) (title, description string) {
	sources, _ := SplitSourceFiles(req.SourceFiles)
	base := ""
	if len(sources) > 0 {
		base = sources[0].GSURI[strings.LastIndex(sources[0].GSURI, "/")+1:]
		stem := base
		if i := strings.LastIndex(stem, "."); i > 0 {
			stem = stem[:i]
		}
		if stem = strings.TrimSpace(stem); stem != "" {
			title = "Test set - " + stem
		}
	}
	if title == "" {
		if snippet := strings.TrimSpace(truncateRunes(strings.TrimSpace(req.AuthorPrompt), 64)); snippet != "" {
			title = "Test set - " + snippet
		} else {
			title = "Generated test set"
		}
	}
	description = fmt.Sprintf("Auto-proposed from %d generated question(s)", len(DraftIDs(req.Candidates)))
	if base != "" {
		description += " grounded on " + base
	}
	description += "."
	return truncateRunes(title, ComposeTitleMaxChars), truncateRunes(description, ComposeDescriptionMaxChars)
}

// NormaliseProposal repairs the model's proposal into a contract-valid
// ProposedTestSet: order a permutation of the candidate draft_ids (unknowns
// dropped, dupes dropped, missing appended in submission order); points
// clamped to 1..100 with the uniform default for missing/invalid;
// title/description clamped with the deterministic fallback when blank.
func NormaliseProposal(parsed map[string]any, req ComposeRequest) ProposedTestSet {
	draftIDs := DraftIDs(req.Candidates)
	known := map[string]bool{}
	for _, d := range draftIDs {
		known[d] = true
	}
	fbTitle, fbDesc := FallbackTitleDescription(req)
	title := truncateRunes(strings.TrimSpace(str(parsed["title"])), ComposeTitleMaxChars)
	if title == "" {
		title = fbTitle
	}
	desc := truncateRunes(strings.TrimSpace(str(parsed["description"])), ComposeDescriptionMaxChars)
	if desc == "" {
		desc = fbDesc
	}
	order := []string{}
	seen := map[string]bool{}
	if rawOrder, ok := parsed["order"].([]any); ok {
		for _, e := range rawOrder {
			did := strings.TrimSpace(str(e))
			if known[did] && !seen[did] {
				order = append(order, did)
				seen[did] = true
			}
		}
	}
	for _, d := range draftIDs {
		if !seen[d] {
			order = append(order, d)
			seen[d] = true
		}
	}
	points := map[string]int{}
	rawPoints, _ := parsed["points"].(map[string]any)
	for _, d := range draftIDs {
		points[d] = coercePoints(rawPoints[d])
	}
	return ProposedTestSet{Title: title, Description: desc, Order: order, Points: points}
}

func coercePoints(v any) int {
	var pts int
	switch t := v.(type) {
	case bool, nil:
		return ComposeDefaultPoints
	case float64:
		pts = int(t)
	case int:
		pts = t
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(t))
		if err != nil {
			return ComposeDefaultPoints
		}
		pts = n
	default:
		return ComposeDefaultPoints
	}
	if pts < ComposePointsMin {
		return ComposePointsMin
	}
	if pts > ComposePointsMax {
		return ComposePointsMax
	}
	return pts
}

// ComposeEnvelope renders the completion output_payload.
func ComposeEnvelope(p ProposedTestSet) string {
	if p.Order == nil {
		p.Order = []string{}
	}
	if p.Points == nil {
		p.Points = map[string]int{}
	}
	b, _ := json.Marshal(map[string]any{"proposed_test_set": p})
	return string(b)
}
