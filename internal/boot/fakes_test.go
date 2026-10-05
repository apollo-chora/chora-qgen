// Test doubles shared across the boot package tests.
//
// fakeReadonlyState + fakeReadonlyContext moved here verbatim from
// cmd/qgen_question/main_test.go when the boot wiring was extracted. They
// implement the minimal session.ReadonlyState + agent.ReadonlyContext
// surface the InstructionProvider closures read from, so a provider can be
// exercised without booting the ADK runtime or a model client. Same
// pattern used in instancedispatch_test.go (chora-adk-common).
//
// fakeLLM is the seam that makes agent construction testable: NewLLMAgent,
// NewQuestionPipeline and NewCriticAgent take adkmodel.LLM, so a test
// passes this and production passes the real modelgatewayclient.
package boot

import (
	"context"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type fakeReadonlyState struct {
	data map[string]any
}

func (f *fakeReadonlyState) Get(k string) (any, error) {
	v, ok := f.data[k]
	if !ok {
		return nil, fmt.Errorf("key %q not found", k)
	}
	return v, nil
}

func (f *fakeReadonlyState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range f.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

// Assert that fakeReadonlyState honours the contract.
var _ session.ReadonlyState = (*fakeReadonlyState)(nil)

type fakeReadonlyContext struct {
	context.Context
	state session.ReadonlyState
}

func (f *fakeReadonlyContext) UserContent() *genai.Content { return nil }
func (f *fakeReadonlyContext) InvocationID() string        { return "test-invocation" }
func (f *fakeReadonlyContext) AgentName() string           { return "qgen_question_generation" }
func (f *fakeReadonlyContext) ReadonlyState() session.ReadonlyState {
	return f.state
}
func (f *fakeReadonlyContext) UserID() string    { return "test-user" }
func (f *fakeReadonlyContext) AppName() string   { return "qgen_question" }
func (f *fakeReadonlyContext) SessionID() string { return "test-session" }
func (f *fakeReadonlyContext) Branch() string    { return "" }

// Assert that fakeReadonlyContext honours the contract.
var _ agent.ReadonlyContext = (*fakeReadonlyContext)(nil)

func newFakeCtx(state map[string]any) *fakeReadonlyContext {
	return &fakeReadonlyContext{
		Context: context.Background(),
		state:   &fakeReadonlyState{data: state},
	}
}

// fakeWritableState satisfies session.State (Get + Set + All) for the
// plugin callbacks, which take agent.CallbackContext rather than the
// readonly one.
type fakeWritableState struct {
	data map[string]any
}

func (f *fakeWritableState) Get(k string) (any, error) {
	v, ok := f.data[k]
	if !ok {
		return nil, fmt.Errorf("key %q not found", k)
	}
	return v, nil
}

func (f *fakeWritableState) Set(k string, v any) error {
	if f.data == nil {
		f.data = map[string]any{}
	}
	f.data[k] = v
	return nil
}

func (f *fakeWritableState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range f.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

var _ session.State = (*fakeWritableState)(nil)

// fakeCallbackContext implements agent.CallbackContext over a writable
// state map. Artifacts is never touched by the boot plugins, so it returns
// the zero value.
type fakeCallbackContext struct {
	*fakeReadonlyContext
	state *fakeWritableState
}

func (f *fakeCallbackContext) Artifacts() agent.Artifacts { return nil }
func (f *fakeCallbackContext) State() session.State       { return f.state }

var _ agent.CallbackContext = (*fakeCallbackContext)(nil)

func newFakeCallbackCtx(state map[string]any) *fakeCallbackContext {
	if state == nil {
		state = map[string]any{}
	}
	ws := &fakeWritableState{data: state}
	return &fakeCallbackContext{
		fakeReadonlyContext: &fakeReadonlyContext{
			Context: context.Background(),
			state:   ws,
		},
		state: ws,
	}
}

// fakeLLM is a non-dialling adkmodel.LLM. It never contacts the gateway;
// GenerateContent is unused by the construction paths under test and
// yields nothing if it is ever driven.
type fakeLLM struct {
	name string
}

func (f *fakeLLM) Name() string { return f.name }

func (f *fakeLLM) GenerateContent(context.Context, *adkmodel.LLMRequest, bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(func(*adkmodel.LLMResponse, error) bool) {}
}

var _ adkmodel.LLM = (*fakeLLM)(nil)
