package boot

import (
	"testing"

	"google.golang.org/adk/plugin"
)

// ---------------------------------------------------------------------------
// StateString
// ---------------------------------------------------------------------------

func TestStateString(t *testing.T) {
	state := &fakeReadonlyState{data: map[string]any{
		"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
		"attempt":     float64(2),
		"nil_value":   nil,
		"empty":       "",
	}}

	cases := []struct {
		name string
		key  string
		want string
	}{
		{"hit returns the string", "traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
		{"miss returns empty", "not_present", ""},
		{"wrong type returns empty", "attempt", ""},
		{"nil value returns empty", "nil_value", ""},
		{"empty string stays empty", "empty", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := StateString(state, tc.key); got != tc.want {
				t.Errorf("StateString(%q) = %q, want %q", tc.key, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Inbound trace headers + plugin
// ---------------------------------------------------------------------------

func TestInboundTraceHeaders(t *testing.T) {
	const tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

	t.Run("both headers present", func(t *testing.T) {
		gotTP, gotTS, ok := InboundTraceHeaders(&fakeReadonlyState{data: map[string]any{
			"traceparent": tp,
			"tracestate":  "chora=orchestrator",
		}})
		if !ok {
			t.Fatal("ok = false with a traceparent in state")
		}
		if gotTP != tp {
			t.Errorf("traceparent = %q, want %q", gotTP, tp)
		}
		if gotTS != "chora=orchestrator" {
			t.Errorf("tracestate = %q, want %q", gotTS, "chora=orchestrator")
		}
	})

	t.Run("traceparent only", func(t *testing.T) {
		gotTP, gotTS, ok := InboundTraceHeaders(&fakeReadonlyState{data: map[string]any{
			"traceparent": tp,
		}})
		if !ok {
			t.Fatal("ok = false with a traceparent in state")
		}
		if gotTP != tp || gotTS != "" {
			t.Errorf("got (%q, %q), want (%q, \"\")", gotTP, gotTS, tp)
		}
	})

	t.Run("no traceparent is a safe skip", func(t *testing.T) {
		// The agent runs on its own trace tree. tracestate alone is
		// meaningless and must not be forwarded.
		_, _, ok := InboundTraceHeaders(&fakeReadonlyState{data: map[string]any{
			"tracestate": "chora=orchestrator",
		}})
		if ok {
			t.Error("ok = true with no traceparent in state")
		}
	})
}

func TestNewInboundTracePlugin(t *testing.T) {
	p, err := NewInboundTracePlugin(CrewKindQuestion)
	if err != nil {
		t.Fatalf("NewInboundTracePlugin: %v", err)
	}
	if p.Name() != "chora_inbound_trace_qgen_question" {
		t.Errorf("plugin name = %q, want %q", p.Name(), "chora_inbound_trace_qgen_question")
	}

	cb := p.BeforeAgentCallback()
	if cb == nil {
		t.Fatal("BeforeAgentCallback is nil; the inbound link would never be attached")
	}

	t.Run("with a traceparent", func(t *testing.T) {
		content, err := cb(newFakeCallbackCtx(map[string]any{
			"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
			"tracestate":  "chora=orchestrator",
		}))
		if err != nil {
			t.Fatalf("callback returned error: %v", err)
		}
		if content != nil {
			t.Error("callback returned content; it must not alter the run")
		}
	})

	t.Run("without a traceparent is a clean no-op", func(t *testing.T) {
		content, err := cb(newFakeCallbackCtx(map[string]any{}))
		if err != nil {
			t.Fatalf("no-op path returned error: %v", err)
		}
		if content != nil {
			t.Error("no-op path returned content; it must not alter the run")
		}
	})

	t.Run("with a malformed traceparent is a clean no-op", func(t *testing.T) {
		content, err := cb(newFakeCallbackCtx(map[string]any{
			"traceparent": "not-a-w3c-header",
		}))
		if err != nil {
			t.Fatalf("malformed traceparent returned error: %v", err)
		}
		if content != nil {
			t.Error("malformed traceparent returned content; it must not alter the run")
		}
	})
}

// ---------------------------------------------------------------------------
// Termination plugin configuration: load-bearing identity strings
// ---------------------------------------------------------------------------

func TestQuestionTerminationConfig_pinsTheCrewIdentity(t *testing.T) {
	cfg := QuestionTerminationConfig()

	if cfg.AgentID != "qgen_question_p2_pipeline" {
		t.Errorf("AgentID = %q, want %q", cfg.AgentID, "qgen_question_p2_pipeline")
	}
	if cfg.CrewKind != CrewKindQuestion {
		t.Errorf("CrewKind = %q, want %q", cfg.CrewKind, CrewKindQuestion)
	}
	if cfg.CrewPattern != "P2_SEQUENTIAL_PIPELINE" {
		t.Errorf("CrewPattern = %q, want %q", cfg.CrewPattern, "P2_SEQUENTIAL_PIPELINE")
	}
	if cfg.Runtime != "AGENT_EXECUTION_RUNTIME_ADK_GO" {
		t.Errorf("Runtime = %q, want %q", cfg.Runtime, "AGENT_EXECUTION_RUNTIME_ADK_GO")
	}
	// 3 sub-agents x 3 iterations, half of qgen_pipeline's 18 cap.
	if cfg.MaxIterations != 9 {
		t.Errorf("MaxIterations = %d, want 9", cfg.MaxIterations)
	}
	if cfg.Publisher == nil {
		t.Error("Publisher is nil; the terminated event would never be emitted")
	}
}

func TestCriticTerminationConfig_pinsTheCrewIdentity(t *testing.T) {
	cfg := CriticTerminationConfig()

	if cfg.AgentID != "qgen_critic_p1_single" {
		t.Errorf("AgentID = %q, want %q", cfg.AgentID, "qgen_critic_p1_single")
	}
	if cfg.CrewKind != CrewKindCritic {
		t.Errorf("CrewKind = %q, want %q", cfg.CrewKind, CrewKindCritic)
	}
	if cfg.CrewPattern != "P1_SINGLE_AGENT" {
		t.Errorf("CrewPattern = %q, want %q", cfg.CrewPattern, "P1_SINGLE_AGENT")
	}
	if cfg.Runtime != "AGENT_EXECUTION_RUNTIME_ADK_GO" {
		t.Errorf("Runtime = %q, want %q", cfg.Runtime, "AGENT_EXECUTION_RUNTIME_ADK_GO")
	}
	// The critic is single-shot per attempt; the orchestrator owns the
	// quality loop's outer iteration.
	if cfg.MaxIterations != 3 {
		t.Errorf("MaxIterations = %d, want 3", cfg.MaxIterations)
	}
	if cfg.Publisher == nil {
		t.Error("Publisher is nil; the terminated event would never be emitted")
	}
}

func TestTerminationConfigs_doNotShareAnAgentID(t *testing.T) {
	// The orchestrator terminal-reads the event author. Two crews sharing
	// one AgentID would make the terminated events indistinguishable.
	if QuestionTerminationConfig().AgentID == CriticTerminationConfig().AgentID {
		t.Error("the two crews share an AgentID")
	}
}

// ---------------------------------------------------------------------------
// Mana plugin
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Plugin chains: ORDER is behaviour
// ---------------------------------------------------------------------------

func TestNewQuestionPlugins_orderAndNames(t *testing.T) {
	plugins, err := NewQuestionPlugins()
	if err != nil {
		t.Fatalf("NewQuestionPlugins: %v", err)
	}
	assertPluginNames(t, plugins, []string{
		"chora_inbound_trace_qgen_question",
		"chora_gateway_tenant_propagation_qgen_question",
		"chora_grounding_qgen_question",
		"chora_termination_qgen_question_p2_pipeline",
	})
}

func TestNewCriticPlugins_orderAndNames(t *testing.T) {
	plugins, err := NewCriticPlugins()
	if err != nil {
		t.Fatalf("NewCriticPlugins: %v", err)
	}
	// The critic grounds on nothing: no groundingplugin in this chain.
	assertPluginNames(t, plugins, []string{
		"chora_inbound_trace_qgen_critic",
		"chora_gateway_tenant_propagation_qgen_critic",
		"chora_termination_qgen_critic_p1_single",
	})
}

func assertPluginNames(t *testing.T, plugins []*plugin.Plugin, want []string) {
	t.Helper()
	var got []string
	for _, p := range plugins {
		if p == nil {
			t.Fatal("plugin chain contains a nil entry")
		}
		got = append(got, p.Name())
	}
	if len(got) != len(want) {
		t.Fatalf("plugin chain = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("plugin %d = %q, want %q (full chain: %v)", i, got[i], want[i], got)
		}
	}
}

// ---------------------------------------------------------------------------
// Launcher config
// ---------------------------------------------------------------------------
