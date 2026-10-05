// Package agentconfig holds the build-time, per-agent model + prompt
// configuration for the qgen crews (AGENT-DRIVEN tiering, CR qgen 2026-06-01).
//
// Each agent owns its OWN embedded YAML (separation of concern) declaring,
// per sub-agent: tier / primary_model / fallback_models / prompt_version. The
// agent reads its config at boot and:
//   - sends primary_model as the gateway logical_model_id,
//   - sends fallback_models as InvokeRequest.fallback_logical_model_ids (the
//     gateway honours the declared chain rather than hardcoding a per-tier
//     ladder),
//   - composes the prompt_version'd template (resolved against the prompt
//     registry, falling back to the embedded v1).
//
// The YAML is the single source of truth; ops may override an individual
// primary via env var for quick experiments, but fallback + tier + prompt
// version stay config-declared.
package agentconfig

import (
	_ "embed"
	"fmt"

	"gopkg.in/yaml.v3"
)

//go:embed qgen_question.yaml
var qgenQuestionYAML []byte

//go:embed qgen_renderer.yaml
var qgenRendererYAML []byte

//go:embed qgen_critic.yaml
var qgenCriticYAML []byte

// SubAgentConfig is one sub-agent's resolved model tier + prompt selection.
type SubAgentConfig struct {
	Tier           string   `yaml:"tier"`
	PrimaryModel   string   `yaml:"primary_model"`
	FallbackModels []string `yaml:"fallback_models"`
	PromptVersion  string   `yaml:"prompt_version"`
}

// AgentConfig is one agent binary's full per-sub-agent config.
type AgentConfig struct {
	Agent     string                    `yaml:"agent"`
	SubAgents map[string]SubAgentConfig `yaml:"sub_agents"`
}

// Sub returns the named sub-agent config, failing loud if the YAML omits it —
// a missing sub-agent is a build/config error, never a silent default (per
// feedback_no_stubs_real_wiring).
func (c AgentConfig) Sub(name string) (SubAgentConfig, error) {
	sc, ok := c.SubAgents[name]
	if !ok {
		return SubAgentConfig{}, fmt.Errorf("agentconfig: agent %q has no sub-agent %q", c.Agent, name)
	}
	if sc.PrimaryModel == "" {
		return SubAgentConfig{}, fmt.Errorf("agentconfig: agent %q sub-agent %q has empty primary_model", c.Agent, name)
	}
	if sc.PromptVersion == "" {
		return SubAgentConfig{}, fmt.Errorf("agentconfig: agent %q sub-agent %q has empty prompt_version", c.Agent, name)
	}
	return sc, nil
}

// QGenQuestion parses the embedded qgen_question config.
func QGenQuestion() (AgentConfig, error) { return parse(qgenQuestionYAML) }

// QGenCritic parses the embedded qgen_critic config.
func QGenCritic() (AgentConfig, error) { return parse(qgenCriticYAML) }

// QGenRenderer parses the embedded qgen_renderer config (the image model).
func QGenRenderer() (AgentConfig, error) { return parse(qgenRendererYAML) }

func parse(raw []byte) (AgentConfig, error) {
	var c AgentConfig
	if err := yaml.Unmarshal(raw, &c); err != nil {
		return AgentConfig{}, fmt.Errorf("agentconfig: unmarshal: %w", err)
	}
	if c.Agent == "" {
		return AgentConfig{}, fmt.Errorf("agentconfig: missing top-level agent name")
	}
	if len(c.SubAgents) == 0 {
		return AgentConfig{}, fmt.Errorf("agentconfig: agent %q declares no sub_agents", c.Agent)
	}
	return c, nil
}
