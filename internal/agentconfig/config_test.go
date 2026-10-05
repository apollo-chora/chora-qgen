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
	assert.Equal(t, "gemini-3.5-flash", assurance.PrimaryModel)
	assert.Equal(t, []string{"gemini-2.5-flash"}, assurance.FallbackModels)
	assert.Equal(t, "v1", assurance.PromptVersion)

	// HIGH tier — generation + evaluation.
	for _, name := range []string{"generation", "evaluation"} {
		sc, err := cfg.Sub(name)
		require.NoError(t, err, name)
		assert.Equal(t, "high", sc.Tier, name)
		assert.Equal(t, "gemini-3.1-pro-preview", sc.PrimaryModel, name)
		assert.Equal(t, []string{"gemini-2.5-pro"}, sc.FallbackModels, name)
	}
}

func TestQGenCritic_CheapTier(t *testing.T) {
	cfg, err := agentconfig.QGenCritic()
	require.NoError(t, err)
	assert.Equal(t, "qgen_critic", cfg.Agent)

	critic, err := cfg.Sub("critic")
	require.NoError(t, err)
	assert.Equal(t, "cheap", critic.Tier)
	assert.Equal(t, "gemini-3.5-flash", critic.PrimaryModel)
	assert.Equal(t, []string{"gemini-2.5-flash"}, critic.FallbackModels)
}

func TestSub_MissingSubAgentFailsLoud(t *testing.T) {
	cfg, err := agentconfig.QGenCritic()
	require.NoError(t, err)
	_, err = cfg.Sub("nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no sub-agent")
}
