// Internal-package tests for the agentconfig fail-loud guards.
//
// The sibling config_test.go is an external (`agentconfig_test`) suite and can
// only reach parse() through the embedded-YAML entry points, both of which
// carry a valid document. The guards below exist precisely so a BROKEN or
// INCOMPLETE config aborts the agent at boot instead of silently defaulting a
// prompt version (feedback_no_stubs_real_wiring), so they need a malformed
// document and a hand-built AgentConfig to be exercised at all.
package agentconfig

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParse_MalformedYAMLFailsLoud(t *testing.T) {
	// An unterminated flow sequence: yaml.Unmarshal must surface, not swallow.
	_, err := parse([]byte("agent: [unterminated\n"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "agentconfig: unmarshal")
}

func TestParse_MissingAgentNameFailsLoud(t *testing.T) {
	// Structurally valid YAML, but with no top-level `agent`. The agent name is
	// what every downstream error message identifies the binary by, so an empty
	// one is a config error rather than an anonymous agent.
	_, err := parse([]byte("sub_agents:\n  critic:\n    tier: cheap\n    prompt_version: v1\n"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing top-level agent name")
}

func TestParse_EmptySubAgentsFailsLoud(t *testing.T) {
	// A named agent that declares nothing to run. Every Sub() call would fail
	// later with a per-sub-agent error, so the config is refused up front.
	_, err := parse([]byte("agent: qgen_probe\nsub_agents: {}\n"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "declares no sub_agents")
	assert.Contains(t, err.Error(), "qgen_probe", "the error must name the offending agent")
}

func TestParse_ValidDocumentRoundTrips(t *testing.T) {
	// Positive control for the three guards above: the same parse() call on a
	// complete document returns the config, so a failure above is the guard
	// firing and not parse() being broken outright.
	cfg, err := parse([]byte("agent: qgen_probe\nsub_agents:\n  critic:\n    tier: cheap\n    prompt_version: v1\n"))
	require.NoError(t, err)
	assert.Equal(t, "qgen_probe", cfg.Agent)
	sub, err := cfg.Sub("critic")
	require.NoError(t, err)
	assert.Equal(t, "cheap", sub.Tier)
	assert.Equal(t, "v1", sub.PromptVersion)
}

func TestSub_EmptyPromptVersionFailsLoud(t *testing.T) {
	// prompt_version selects the prompt-registry template. Blank means the
	// composed prompt is unversioned, which breaks the ADR-197 stamping record.
	cfg := AgentConfig{
		Agent: "qgen_probe",
		SubAgents: map[string]SubAgentConfig{
			"generation": {Tier: "high"},
		},
	}
	_, err := cfg.Sub("generation")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty prompt_version")
}
