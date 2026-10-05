// Package tools — qgen_question ADK tool surface per ADR-153.
//
// The parse_document tool is the multimodal-extraction surface registered on
// the qgen_question_assurance agent. Per ADR-153 §"chora-doc-parser as ADK
// tool" it is a **fail-loud stub at M14.0** — returning a canonical error
// envelope unconditionally — so that:
//
//  1. The LLM's tool-calling loop reaches a deterministic refusal when an
//     author attaches a file before chora-doc-parser ships (M14.1).
//  2. There is NO fake-success path: per the no-stubs-real-wiring rule,
//     stub responses MUST surface as errors so callers cannot silently
//     accumulate fabricated parse output.
//  3. The wire shape stays stable across the M14.0 → M14.1 cutover — at
//     M14.1 the handler body swaps to a gRPC client call; no agent prompt
//     change needed.
//
// When the LLM never invokes the tool (the dominant text-only single-Q path),
// the binding is dormant and adds zero latency.
package tools

import (
	"context"
	"log/slog"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// ParseDocumentInput is the canonical input shape for the parse_document
// tool. The Format hint helps chora-doc-parser pick the deterministic
// extractor (M14.1); at M14.0 the handler ignores the input entirely.
type ParseDocumentInput struct {
	FileURI string `json:"file_uri"`
	Format  string `json:"format"` // "pdf" | "docx" | "md" | "txt"
}

// ParseDocumentOutput is the canonical output shape. Per ADR-153 §
// "chora-doc-parser as ADK tool" the structure stays stable across the
// M14.0 stub → M14.1 real-wiring cutover. Fields populated:
//
//   - M14.0 stub:  Error="doc_parser_not_yet_wired" + Reason=<defer note>
//   - M14.1 real:  Text, Images, ParsedAt; Error/Reason empty
//
// NEVER return both a successful Text and an Error in the same envelope.
type ParseDocumentOutput struct {
	Text     string             `json:"text,omitempty"`
	Images   []ImageDescription `json:"images,omitempty"`
	ParsedAt string             `json:"parsed_at,omitempty"`
	Error    string             `json:"error,omitempty"`
	Reason   string             `json:"reason,omitempty"`
}

// ImageDescription is one image extracted from a parsed document. M14.1 the
// chora-doc-parser deterministic extractor populates ObjectURI + AltText;
// LLMDescription is filled by the multimodal-capable assurance LLM in a
// follow-up sub-call.
type ImageDescription struct {
	ObjectURI      string `json:"object_uri,omitempty"`
	AltText        string `json:"alt_text,omitempty"`
	LLMDescription string `json:"llm_description,omitempty"`
}

// ParseDocumentHandler is the ADK tool.Func handler for parse_document.
// At M14.0 it returns the canonical fail-loud envelope unconditionally
// per ADR-153 — chora-doc-parser is the M14.1 deliverable per
// docs/architecture/qgen-crew-composition-2026-05-12.md §9 item 7.
//
// The slog log line carries crew + step attribution so the dormant
// invocations stay correlatable with the rest of the qgen_question trace
// tree.
func ParseDocumentHandler(_ tool.Context, in ParseDocumentInput) (ParseDocumentOutput, error) {
	slog.Default().LogAttrs(context.Background(), slog.LevelInfo,
		"parse_document_tool_invoked_stub",
		slog.String("chora.crew_kind", "qgen_question"),
		slog.String("chora.qgen.step", "qgen_question_assurance"),
		slog.String("file_uri", in.FileURI),
		slog.String("format", in.Format),
	)
	return ParseDocumentOutput{
		Error:  "doc_parser_not_yet_wired",
		Reason: "M14.1 deliverable per docs/architecture/qgen-crew-composition-2026-05-12.md §9",
	}, nil
}

// NewParseDocumentTool constructs the ADK functiontool wrapper around
// ParseDocumentHandler. The Name + Description are visible to the LLM
// at every assurance turn (advertised in the assurance prompt — see
// composer_question.go StepAssurance3()).
func NewParseDocumentTool() (tool.Tool, error) {
	return functiontool.New(functiontool.Config{
		Name: "parse_document",
		Description: "Extract text + image descriptions from an attached document. " +
			"Supports pdf / docx / md / txt. M14.0 fail-loud stub — returns " +
			"{error: doc_parser_not_yet_wired} until chora-doc-parser ships at M14.1. " +
			"Invoke ONLY when the author has attached a file; dormant for text-only single-Q.",
	}, ParseDocumentHandler)
}
