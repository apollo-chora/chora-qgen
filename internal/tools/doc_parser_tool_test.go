package tools

import (
	"testing"
)

// The parse_document tool is a fail-loud stub at M14.0 per ADR-153. Every
// invocation MUST return the canonical error envelope — no fake-success
// path. Per the no-stubs-real-wiring rule, the test pins this contract.

func TestParseDocumentHandler_returnsFailLoudEnvelopeForAllFormats(t *testing.T) {
	cases := []ParseDocumentInput{
		{FileURI: "s3://bucket/doc.pdf", Format: "pdf"},
		{FileURI: "s3://bucket/doc.docx", Format: "docx"},
		{FileURI: "s3://bucket/doc.md", Format: "md"},
		{FileURI: "s3://bucket/doc.txt", Format: "txt"},
		{FileURI: "", Format: ""}, // empty input — still fail-loud
	}
	wantErr := "doc_parser_not_yet_wired"
	wantReason := "M14.1 deliverable per docs/architecture/qgen-crew-composition-2026-05-12.md §9"
	for _, in := range cases {
		out, err := ParseDocumentHandler(nil, in)
		if err != nil {
			t.Errorf("input=%+v: handler returned error %v; want nil (the failure is in the envelope, not the Go error)", in, err)
		}
		if out.Error != wantErr {
			t.Errorf("input=%+v: Error=%q; want %q", in, out.Error, wantErr)
		}
		if out.Reason != wantReason {
			t.Errorf("input=%+v: Reason=%q; want %q", in, out.Reason, wantReason)
		}
		// MUST NOT carry a fake-success envelope.
		if out.Text != "" {
			t.Errorf("input=%+v: fail-loud stub leaked Text=%q (no fake-success path allowed)", in, out.Text)
		}
		if len(out.Images) > 0 {
			t.Errorf("input=%+v: fail-loud stub leaked %d Images (no fake-success path allowed)", in, len(out.Images))
		}
		if out.ParsedAt != "" {
			t.Errorf("input=%+v: fail-loud stub leaked ParsedAt=%q (no fake-success path allowed)", in, out.ParsedAt)
		}
	}
}

func TestNewParseDocumentTool_buildsCleanly(t *testing.T) {
	tt, err := NewParseDocumentTool()
	if err != nil {
		t.Fatalf("NewParseDocumentTool: %v", err)
	}
	if tt.Name() != "parse_document" {
		t.Errorf("tool.Name() = %q; want %q", tt.Name(), "parse_document")
	}
	if tt.Description() == "" {
		t.Error("tool.Description() is empty — the LLM must see the M14.0 stub posture")
	}
}
