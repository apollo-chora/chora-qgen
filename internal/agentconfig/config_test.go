package agentconfig_test

import (
	"testing"

	"github.com/apollo-chora/chora-qgen/internal/agentconfig"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQGenQuestion_TierLadders(t *testing.T) {
	cfg, err := agentconfig.QGenQuestion()
	require.NoError(t, err)
	assert.Equal(t, "qgen_question", cfg.Agent)

	// CHEAP tier — assurance.
	assurance, err := cfg.Sub("assurance")
	require.NoError(t, err)
	assert.Equal(t, "cheap", assurance.Tier)
	assert.Equal(t, "v1", assurance.PromptVersion)

	// HIGH tier — generation + evaluation.
	for _, name := range []string{"generation", "evaluation"} {
		sc, err := cfg.Sub(name)
		require.NoError(t, err, name)
		assert.Equal(t, "high", sc.Tier, name)
		assert.Equal(t, "v1", sc.PromptVersion, name)
	}

	// The compose sub-agent (qgen_generate mode=compose) is declared too.
	compose, err := cfg.Sub("compose")
	require.NoError(t, err)
	assert.Equal(t, "high", compose.Tier)
	assert.Equal(t, "v1", compose.PromptVersion)
}

func TestQGenCritic_CheapTier(t *testing.T) {
	cfg, err := agentconfig.QGenCritic()
	require.NoError(t, err)
	assert.Equal(t, "qgen_critic", cfg.Agent)

	critic, err := cfg.Sub("critic")
	require.NoError(t, err)
	assert.Equal(t, "cheap", critic.Tier)
	assert.Equal(t, "v1", critic.PromptVersion)
}

func TestQGenRenderer_ImageTier(t *testing.T) {
	cfg, err := agentconfig.QGenRenderer()
	require.NoError(t, err)
	assert.Equal(t, "qgen_renderer", cfg.Agent)

	renderer, err := cfg.Sub("renderer")
	require.NoError(t, err)
	assert.Equal(t, "image", renderer.Tier)
	assert.Equal(t, "v1", renderer.PromptVersion)
}

func TestSub_MissingSubAgentFailsLoud(t *testing.T) {
	cfg, err := agentconfig.QGenCritic()
	require.NoError(t, err)
	_, err = cfg.Sub("nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no sub-agent")
}
