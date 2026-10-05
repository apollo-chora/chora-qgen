package boot

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"

	"google.golang.org/adk/agent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/apollo-chora/chora-adk-common/agentdispatch"
)

func composeCandidates() []any {
	return []any{
		map[string]any{"draft_id": "d-1", "stem": "What is 3/6 as a fraction in its simplest form?", "question_type": "mcq"},
		map[string]any{"draft_id": "d-2", "stem": "Explain why 1/2 = 2/4.", "question_type": "oe"},
		map[string]any{"draft_id": "d-3", "stem": strings.Repeat("long ", 200), "question_type": "mcq"},
		map[string]any{"no_draft": true},
	}
}

func composeState() map[string]any {
	return map[string]any{
		"tenant_id": "11111111-1111-7111-8111-111111111111", "user_gcid": "g1", "mode": "compose",
		"candidates":    composeCandidates(),
		"author_prompt": "Primary 5 fractions end-of-term paper, 3 questions",
		"metadata":      map[string]any{"batch_id": "b-1"},
		"source_files": []any{
			map[string]any{"gs_uri": "gs://chora-creation-uploads/t/paper-2024.pdf", "mime_type": "application/pdf", "role": "source"},
			map[string]any{"gs_uri": "gs://chora-creation-uploads/t/mark-scheme.pdf", "mime_type": "application/pdf; charset=binary", "role": "rubric"},
			map[string]any{"gs_uri": "gs://chora-creation-uploads/t/second-rubric.pdf", "mime_type": "application/pdf", "role": "rubric"},
		},
		"grounding_mode": "strict",
	}
}

func TestParseComposeRequestAndSplit(t *testing.T) {
	req, err := ParseComposeRequest(mapState(composeState()))
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Candidates) != 4 || req.AuthorPrompt == "" || req.GroundingMod != "strict" || len(req.SourceFiles) != 3 {
		t.Fatalf("request = %+v", req)
	}
	sources, rubric := SplitSourceFiles(req.SourceFiles)
	if len(sources) != 1 || rubric == nil || rubric.GSURI != "gs://chora-creation-uploads/t/mark-scheme.pdf" {
		t.Fatalf("split: sources=%v rubric=%v (first rubric wins, the second is dropped)", sources, rubric)
	}
	if ids := DraftIDs(req.Candidates); len(ids) != 3 || ids[2] != "d-3" {
		t.Fatalf("draft ids = %v", ids)
	}
	// candidates as a JSON string also decode; a non-list is permanent.
	st := composeState()
	st["candidates"] = `[{"draft_id":"x"}]`
	if req, err := ParseComposeRequest(mapState(st)); err != nil || len(req.Candidates) != 1 {
		t.Fatalf("string candidates: %+v %v", req, err)
	}
	var perm *agentdispatch.PermanentError
	st["candidates"] = "not json"
	if _, err := ParseComposeRequest(mapState(st)); !errors.As(err, &perm) || !strings.Contains(err.Error(), "invalid_candidates") {
		t.Fatalf("bad candidates: %v", err)
	}
	st = composeState()
	delete(st, "candidates")
	if _, err := ParseComposeRequest(mapState(st)); !errors.As(err, &perm) || !strings.Contains(err.Error(), "missing_candidates") {
		t.Fatalf("missing candidates: %v", err)
	}
	st = composeState()
	st["source_files"] = []any{map[string]any{"gs_uri": "https://x/y.pdf"}}
	if _, err := ParseComposeRequest(mapState(st)); !errors.As(err, &perm) || !strings.Contains(err.Error(), "invalid_source_files") {
		t.Fatalf("non-gs source: %v", err)
	}
	st = composeState()
	delete(st, "source_files")
	if req, err := ParseComposeRequest(mapState(st)); err != nil || len(req.SourceFiles) != 0 {
		t.Fatalf("ungrounded batch must parse: %+v %v", req, err)
	}
}

func TestComposeInstructionAndParts(t *testing.T) {
	req, _ := ParseComposeRequest(mapState(composeState()))
	text := ComposeInstructionText(req)
	for _, want := range []string{"Compose ONE test set", "Author's batch request: Primary 5 fractions end-of-term paper", `"draft_id":"d-1"`, `"question_type":"oe"`, `"order": ["<draft_id>", ...]`, "max 256 chars", "int 1..100", "use 10."} {
		if !strings.Contains(text, want) {
			t.Fatalf("instruction lacks %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, strings.Repeat("long ", 120)) {
		t.Fatal("stems must be truncated to 512 chars")
	}
	parts := ComposeUserParts(req)
	if len(parts) != 4 || parts[0].Text == "" || parts[1].FileData == nil || parts[1].FileData.FileURI != "gs://chora-creation-uploads/t/paper-2024.pdf" || parts[1].FileData.MIMEType != "application/pdf" {
		t.Fatalf("parts = %+v", parts)
	}
	if parts[2].Text != ComposeRubricMarker || parts[3].FileData == nil || parts[3].FileData.FileURI != "gs://chora-creation-uploads/t/mark-scheme.pdf" || parts[3].FileData.MIMEType != "application/pdf" {
		t.Fatalf("rubric must follow its marker with a clean mime: %+v %+v", parts[2], parts[3])
	}
}

func TestParseComposeAnswerAndNormalise(t *testing.T) {
	req, _ := ParseComposeRequest(mapState(composeState()))
	parsed, err := ParseComposeAnswer("Sure:\n```json\n{\"title\": \"  Fractions paper 2024 \", \"order\": [\"d-2\", \"d-9\", \"d-2\", \"d-1\"], \"points\": {\"d-1\": 5, \"d-2\": \"7\", \"d-3\": true, \"d-9\": 3}}\n```\nthanks")
	if err != nil {
		t.Fatal(err)
	}
	p := NormaliseProposal(parsed, req)
	if p.Title != "Fractions paper 2024" {
		t.Fatalf("title = %q", p.Title)
	}
	if !strings.HasPrefix(p.Description, "Auto-proposed from 3 generated question(s) grounded on paper-2024.pdf") {
		t.Fatalf("blank description must take the deterministic fallback: %q", p.Description)
	}
	if strings.Join(p.Order, ",") != "d-2,d-1,d-3" {
		t.Fatalf("order = %v (unknown dropped, dupe dropped, missing appended)", p.Order)
	}
	if p.Points["d-1"] != 5 || p.Points["d-2"] != 7 || p.Points["d-3"] != ComposeDefaultPoints || len(p.Points) != 3 {
		t.Fatalf("points = %v", p.Points)
	}
	// Clamps and bools.
	p = NormaliseProposal(map[string]any{"points": map[string]any{"d-1": 0, "d-2": 500, "d-3": false}, "title": strings.Repeat("t", 300)}, req)
	if p.Points["d-1"] != 1 || p.Points["d-2"] != 100 || p.Points["d-3"] != 10 || len(p.Title) != 256 {
		t.Fatalf("clamps: %+v", p)
	}
	// Fallback titles: prompt snippet when ungrounded, the fixed text when nothing.
	st := composeState()
	delete(st, "source_files")
	ureq, _ := ParseComposeRequest(mapState(st))
	title, desc := FallbackTitleDescription(ureq)
	if !strings.HasPrefix(title, "Test set - Primary 5 fractions") || desc != "Auto-proposed from 3 generated question(s)." {
		t.Fatalf("ungrounded fallback = %q / %q", title, desc)
	}
	st["author_prompt"] = " "
	ureq, _ = ParseComposeRequest(mapState(st))
	if title, _ := FallbackTitleDescription(ureq); title != "Generated test set" {
		t.Fatalf("empty prompt fallback = %q", title)
	}
	// Unusable answers.
	for _, bad := range []string{"", "   ", "I cannot compose this.", "[1,2,3]"} {
		if _, err := ParseComposeAnswer(bad); !errors.Is(err, ErrComposeUnusable) {
			t.Fatalf("%q must be unusable: %v", bad, err)
		}
	}
	var env map[string]any
	if err := json.Unmarshal([]byte(ComposeEnvelope(p)), &env); err != nil || env["proposed_test_set"] == nil {
		t.Fatalf("envelope: %v %s", err, ComposeEnvelope(p))
	}
}

// ---- runner harness for the root router ----

type scriptedComposeLLM struct {
	answers []string
	calls   []*adkmodel.LLMRequest
}

func (s *scriptedComposeLLM) Name() string { return "scripted" }
func (s *scriptedComposeLLM) GenerateContent(_ context.Context, req *adkmodel.LLMRequest, _ bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		s.calls = append(s.calls, req)
		if len(s.answers) == 0 {
			yield(nil, errors.New("no scripted answer"))
			return
		}
		text := s.answers[0]
		s.answers = s.answers[1:]
		yield(&adkmodel.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}}}, nil)
	}
}

func runQuestionRoot(t *testing.T, root agent.Agent, state map[string]any) (string, error) {
	t.Helper()
	ctx := context.Background()
	svc := session.InMemoryService()
	const app, user, sid = "qgen_question", "11111111-1111-7111-8111-111111111111:g1", "dispatch:e-1"
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: sid, State: state}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: app, Agent: root, SessionService: svc})
	if err != nil {
		t.Fatal(err)
	}
	var events []*session.Event
	for ev, err := range r.Run(ctx, user, sid, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "BEGIN"}}}, agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		events = append(events, ev)
	}
	return agentdispatch.TerminalText(events, agentdispatch.TerminalAuthor(DispatchRoleGenerate))
}

func TestQuestionRoot_composeRunsTheComposerAndNormalises(t *testing.T) {
	cfg := Config{Binary: CrewKindQuestion, PromptVersion: "v1", ComposeModel: "gemini-2.5-pro", ComposePromptVersion: "v1"}
	gen := &scriptedComposeLLM{answers: []string{"never"}}
	comp := &scriptedComposeLLM{answers: []string{`{"title":"Fractions paper","description":"Three questions on equivalence.","order":["d-3","d-1"],"points":{"d-1":4}}`}}
	root, err := NewQuestionRoot(cfg, gen, comp)
	if err != nil {
		t.Fatal(err)
	}
	if root.Name() != CrewKindQuestion || len(root.SubAgents()) != 2 || root.SubAgents()[0].Name() != questionPipelineName {
		t.Fatalf("tree = %v", root)
	}
	out, err := runQuestionRoot(t, root, composeState())
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		P ProposedTestSet `json:"proposed_test_set"`
	}
	if err := json.Unmarshal([]byte(out), &env); err != nil {
		t.Fatalf("envelope: %v %s", err, out)
	}
	if env.P.Title != "Fractions paper" || strings.Join(env.P.Order, ",") != "d-3,d-1,d-2" || env.P.Points["d-1"] != 4 || env.P.Points["d-2"] != 10 || env.P.Points["d-3"] != 10 {
		t.Fatalf("normalised = %+v", env.P)
	}
	if len(gen.calls) != 0 || len(comp.calls) != 1 {
		t.Fatalf("compose must call only the composer: gen=%d comp=%d", len(gen.calls), len(comp.calls))
	}
	req := comp.calls[0]
	if req.Config == nil || req.Config.SystemInstruction == nil || !strings.Contains(req.Config.SystemInstruction.Parts[0].Text, "assessment designer") {
		t.Fatalf("system prompt missing: %+v", req.Config)
	}
	var user *genai.Content
	for i := len(req.Contents) - 1; i >= 0; i-- {
		if req.Contents[i].Role == "user" {
			user = req.Contents[i]
			break
		}
	}
	if user == nil || len(user.Parts) != 4 || !strings.Contains(user.Parts[0].Text, "Compose ONE test set") || user.Parts[1].FileData == nil || user.Parts[2].Text != ComposeRubricMarker || user.Parts[3].FileData == nil {
		t.Fatalf("user parts = %+v", user)
	}
}

func TestQuestionRoot_permanentFaults(t *testing.T) {
	cfg := Config{Binary: CrewKindQuestion, PromptVersion: "v1", ComposeModel: "gemini-2.5-pro", ComposePromptVersion: "v1"}
	var perm *agentdispatch.PermanentError
	// Unusable composer answer.
	root, _ := NewQuestionRoot(cfg, &scriptedComposeLLM{}, &scriptedComposeLLM{answers: []string{"I would rather not."}})
	if _, err := runQuestionRoot(t, root, composeState()); !errors.As(err, &perm) || !strings.Contains(err.Error(), "compose_unusable_answer") {
		t.Fatalf("unusable answer: %v", err)
	}
	// Unknown mode before any model call.
	gen, comp := &scriptedComposeLLM{answers: []string{"x"}}, &scriptedComposeLLM{answers: []string{"x"}}
	root, _ = NewQuestionRoot(cfg, gen, comp)
	st := composeState()
	st["mode"] = "render"
	if _, err := runQuestionRoot(t, root, st); !errors.As(err, &perm) || !strings.Contains(err.Error(), "unknown_mode: render") || len(gen.calls)+len(comp.calls) != 0 {
		t.Fatalf("unknown mode: %v", err)
	}
	// Compose without a composer model is refused by name, never routed to generation.
	root, _ = NewQuestionRoot(Config{Binary: CrewKindQuestion, PromptVersion: "v1"}, gen, nil)
	if len(root.SubAgents()) != 1 {
		t.Fatalf("no composer: %d sub-agents", len(root.SubAgents()))
	}
	if _, err := runQuestionRoot(t, root, composeState()); !errors.As(err, &perm) || !strings.Contains(err.Error(), "unknown_mode: compose") || len(gen.calls) != 0 {
		t.Fatalf("compose without composer: %v", err)
	}
	// Missing candidates is permanent before the model.
	root, _ = NewQuestionRoot(cfg, gen, comp)
	st = composeState()
	delete(st, "candidates")
	if _, err := runQuestionRoot(t, root, st); !errors.As(err, &perm) || !strings.Contains(err.Error(), "missing_candidates") || len(comp.calls) != 0 {
		t.Fatalf("missing candidates: %v", err)
	}
}

func TestComposeConfigAndGateway(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ComposeModel != "gemini-2.5-pro" || len(cfg.ComposeFallbackModels) != 1 || cfg.ComposeFallbackModels[0] != "gemini-2.5-flash" || cfg.ComposePromptVersion != "v1" {
		t.Fatalf("compose config = %+v", cfg)
	}
	gw := cfg.ComposeGatewayConfig()
	if gw.LogicalModelID != "gemini-2.5-pro" || gw.CrewKind != "qgen_question.compose" || gw.Surface != "qgen" || gw.AgentID != "qgen_question" {
		t.Fatalf("compose gateway = %+v", gw)
	}
	t.Setenv("QGEN_QUESTION_COMPOSE_MODEL", "gemini-3.1-pro-preview")
	cfg, _ = LoadQuestionConfig()
	if cfg.ComposeModel != "gemini-3.1-pro-preview" || cfg.ComposeFallbackModels[0] != "gemini-2.5-flash" {
		t.Fatalf("override replaces the primary only: %+v", cfg)
	}
}
