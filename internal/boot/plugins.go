package boot

import (
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"

	"google.golang.org/genai"

	"github.com/apollo-chora/chora-adk-common/groundingplugin"
	"github.com/apollo-chora/chora-adk-common/modelgatewayclient"
	"github.com/apollo-chora/chora-adk-common/terminationplugin"
	"github.com/apollo-chora/chora-adk-common/tracing"
)

// StateString reads a string value from ADK session state. Returns "" on miss
// or type mismatch (the defensive shape every plugin in the fleet uses).
func StateString(state interface {
	Get(string) (any, error)
}, key string) string {
	v, err := state.Get(key)
	if err != nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// InboundTraceHeaders reads the W3C trace headers the Python LangGraph
// orchestrator (qgen_crew.py via reasoning_engine_executor._build_session_state)
// stamps into ADK session state.
//
// ok is false when there is no traceparent: the agent then runs on its own
// trace tree, which is a safe no-op. A tracestate without a traceparent is
// meaningless and is never forwarded on its own.
func InboundTraceHeaders(state interface {
	Get(string) (any, error)
}) (traceparent, tracestate string, ok bool) {
	traceparent = StateString(state, "traceparent")
	if traceparent == "" {
		return "", "", false
	}
	return traceparent, StateString(state, "tracestate"), true
}

// NewInboundTracePlugin builds the ADK plugin that links an agent's local
// trace to the orchestrator's inbound W3C traceparent (CR qgen Phase B2). It
// reads "traceparent" / "tracestate" from session state in a BeforeAgentCallback
// which is the earliest agent-side hook with both the active invoke_agent span and
// session.State() (it runs after agent.go:StartInvokeAgentSpan), and calls
// tracing.AddInboundLink to annotate the root span. Missing/malformed
// traceparent is a safe no-op.
//
// LIMITATION: the ADK runtime creates the invoke_agent root span from the
// inbound HTTP request context BEFORE any callback fires, and the REST
// server does NOT extract a propagator carrier from the request, so the
// root span cannot be RE-PARENTED across the orchestrator to agent boundary
// from agent code. A span LINK is the best available linkage; the collector
// connects the two trace trees bidirectionally. See tracing/inbound.go.
func NewInboundTracePlugin(crewKind string) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "chora_inbound_trace_" + crewKind,
		BeforeAgentCallback: func(ctx agent.CallbackContext) (*genai.Content, error) {
			traceparent, tracestate, ok := InboundTraceHeaders(ctx.State())
			if !ok {
				return nil, nil
			}
			// ctx embeds context.Context and carries the active ADK
			// invoke_agent span (BeforeAgentCallback runs after
			// StartInvokeAgentSpan). AddInboundLink annotates that span
			// with a link to the orchestrator span.
			tracing.AddInboundLink(ctx, traceparent, tracestate)
			return nil, nil
		},
	})
}

// QuestionTerminationConfig is the qgen_question crew's termination-plugin
// configuration. The plugin emits chora.ai_kernel.agent.terminated.v1 once
// per crew run.
//
// AgentID is the crew identity, not a per-sub-agent one: per-sub-agent
// termination would require a per-llmagent plugin attachment (M14.2+ scope).
// MaxIterations = 9, that is 3 sub-agents x 3 iterations (per-agent average),
// half of qgen_pipeline's 18 cap. The slim-crew win shows up here too.
func QuestionTerminationConfig() terminationplugin.Config {
	return terminationplugin.Config{
		Publisher:     &terminationplugin.LoggingPublisher{},
		AgentID:       "qgen_question_p2_pipeline",
		Runtime:       "AGENT_EXECUTION_RUNTIME_ADK_GO",
		CrewKind:      CrewKindQuestion,
		CrewPattern:   "P2_SEQUENTIAL_PIPELINE",
		MaxIterations: 9,
	}
}

// CriticTerminationConfig is the qgen_critic crew's termination-plugin
// configuration. It emits chora.ai_kernel.agent.terminated.v1 once per critic
// invocation; the D6 P3 trace-emission contract is satisfied via the plugin's
// auto-attached OTel span.
//
// MaxIterations = 3: the critic is single-shot per attempt, and the
// orchestrator owns the quality loop's outer iteration.
func CriticTerminationConfig() terminationplugin.Config {
	return terminationplugin.Config{
		Publisher:     &terminationplugin.LoggingPublisher{},
		AgentID:       "qgen_critic_p1_single",
		Runtime:       "AGENT_EXECUTION_RUNTIME_ADK_GO",
		CrewKind:      CrewKindCritic,
		CrewPattern:   "P1_SINGLE_AGENT",
		MaxIterations: 3,
	}
}

// NewQuestionPlugins builds the qgen_question plugin chain IN ORDER. Order is
// behaviour: the ADK runner invokes each hook in slice order, so the inbound
// trace link is established before any plugin can refuse a run, and the
// tenant propagation stamps the request before grounding appends to it.
//
//	inbound trace  = link the orchestrator's trace (CR qgen Phase B2)
//	mana gate      = token-budget quota refusal (ADR-142)
//	tenant prop    = per-request tenant_id/user_gcid onto the gateway
//	                 LLMRequest so RLS + ledger + budget attribute to the
//	                 REQUESTING tenant, not the process-fixed env tenant
//	                 (ADR-169)
//	grounding      = EPIC-1a: inject the uploaded source material (an
//	                 object-store URI) as a Gemini FileData part. No-op
//	                 when nothing was uploaded, so the single-question and
//	                 non-grounded batch paths are unchanged
//	termination    = emit chora.ai_kernel.agent.terminated.v1
func NewQuestionPlugins() ([]*plugin.Plugin, error) {
	inboundTraceP, err := NewInboundTracePlugin(CrewKindQuestion)
	if err != nil {
		return nil, fmt.Errorf("inbound-trace plugin.New: %w", err)
	}
	tenantPropP, err := modelgatewayclient.NewTenantPropagationPlugin(CrewKindQuestion)
	if err != nil {
		return nil, fmt.Errorf("modelgatewayclient.NewTenantPropagationPlugin: %w", err)
	}
	groundingP, err := groundingplugin.New(CrewKindQuestion)
	if err != nil {
		return nil, fmt.Errorf("groundingplugin.New: %w", err)
	}
	terminationP, err := terminationplugin.New(QuestionTerminationConfig())
	if err != nil {
		return nil, fmt.Errorf("terminationplugin.New: %w", err)
	}
	return []*plugin.Plugin{inboundTraceP, tenantPropP, groundingP, terminationP}, nil
}

// NewCriticPlugins builds the qgen_critic plugin chain IN ORDER. Same shape as
// NewQuestionPlugins minus grounding: the critic reads the candidate out of
// session state and never grounds on uploaded material.
func NewCriticPlugins() ([]*plugin.Plugin, error) {
	inboundTraceP, err := NewInboundTracePlugin(CrewKindCritic)
	if err != nil {
		return nil, fmt.Errorf("inbound-trace plugin.New: %w", err)
	}
	tenantPropP, err := modelgatewayclient.NewTenantPropagationPlugin(CrewKindCritic)
	if err != nil {
		return nil, fmt.Errorf("modelgatewayclient.NewTenantPropagationPlugin: %w", err)
	}
	terminationP, err := terminationplugin.New(CriticTerminationConfig())
	if err != nil {
		return nil, fmt.Errorf("terminationplugin.New: %w", err)
	}
	return []*plugin.Plugin{inboundTraceP, tenantPropP, terminationP}, nil
}

// RendererTerminationConfig pins the renderer's termination identity: one
// image Invoke per dispatch, no ADK model call (MaxIterations is a guard the
// renderer never reaches).
func RendererTerminationConfig() terminationplugin.Config {
	return terminationplugin.Config{
		Publisher:     &terminationplugin.LoggingPublisher{},
		AgentID:       CrewKindRenderer,
		Runtime:       "AGENT_EXECUTION_RUNTIME_ADK_GO",
		CrewKind:      CrewSurface,
		CrewPattern:   "P1_SINGLE_AGENT",
		MaxIterations: 2,
	}
}

// NewRendererPlugins is the renderer chain: [inboundTrace, termination]. No
// tenant-propagation plugin: the renderer makes no ADK model call; it stamps
// tenant, gcid, surface and the dispatch key on its image Invoke itself.
func NewRendererPlugins() ([]*plugin.Plugin, error) {
	inboundTraceP, err := NewInboundTracePlugin(CrewKindRenderer)
	if err != nil {
		return nil, fmt.Errorf("inbound-trace plugin.New: %w", err)
	}
	terminationP, err := terminationplugin.New(RendererTerminationConfig())
	if err != nil {
		return nil, fmt.Errorf("terminationplugin.New: %w", err)
	}
	return []*plugin.Plugin{inboundTraceP, terminationP}, nil
}
