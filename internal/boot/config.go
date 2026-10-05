// Package boot holds the qgen binaries' boot wiring: environment
// resolution, agent construction and plugin construction, extracted out of
// cmd/qgen_question and cmd/qgen_critic so every decision the binaries make
// at start-up is unit-testable.
//
// Split of responsibility:
//
//	config.go  = env plus agentconfig YAML resolution into a Config.
//	             Returns errors, never exits the process.
//	agents.go  = per-turn instruction providers, llmagent + pipeline
//	             construction. The LLM is INJECTED as adkmodel.LLM so a
//	             test can pass a fake and production passes the real
//	             modelgatewayclient.
//	plugins.go = the ADK plugin chain, so the load-bearing identity strings
//	             (AgentID, CrewKind, CrewPattern, MaxIterations) and the
//	             plugin ORDER are assertable.
//	run.go     = the thin glue that dials the outside world (tracing.Init,
//	             modelgatewayclient.New, the object-store client, the blocking
//	             agentdispatch subscriber) behind package-level seams.
//
// Boot failures return a descriptive error. main() is the only place that
// turns one into a non-zero exit.
package boot

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/apollo-chora/chora-adk-common/modelgatewayclient"

	"github.com/apollo-chora/chora-qgen/internal/agentconfig"
)

// Crew identities. These strings are load-bearing: the Python orchestrator's
// executor terminal-reads the event author, the gateway's routing policy
// keys off AgentID, and the plugin names embed the crew kind. Renaming one
// breaks the orchestrator, not just this binary.
const (
	// CrewKindQuestion is the qgen_question crew identity. It doubles as
	// the binary name, the tracing service name and the gateway AgentID.
	CrewKindQuestion = "qgen_question"

	// CrewKindCritic is the qgen_critic crew identity, with the same
	// triple duty as CrewKindQuestion.
	CrewKindCritic = "qgen_critic"
	// CrewKindRenderer is the qgen_renderer identity (ADR-254 D2: the third
	// qgen member, the generative scene image per chunk on role qgen_render).
	CrewKindRenderer = "qgen_renderer"
	// CrewSurface is the crew id stamped as InvokeRequest.surface on every
	// gateway call of the three binaries (ADR-254 D7).
	CrewSurface = "qgen"
)

// Config is one binary's fully resolved boot configuration: the env vars it
// read plus the sub-agent slice of its embedded agentconfig YAML.
//
// Nothing here is a default invented at the call site. Every value comes
// from an env var or from the YAML per feedback_no_inline_config.
type Config struct {
	// Binary is the crew identity of the binary this Config belongs to,
	// CrewKindQuestion or CrewKindCritic. It is also the tracing service
	// name and the boot log message prefix.
	Binary string

	// ModelLabel is the sub-agent label the boot log keys are built from,
	// "generation" for qgen_question and "critic" for qgen_critic. It
	// keeps each binary's log field names byte-identical to the shape
	// they had while the wiring lived in main().
	ModelLabel string

	ProjectID    string
	AgentAppName string

	GatewayEndpoint string
	// GatewayAudience is the ID-token audience claim minted for the gateway
	// call. Read here rather than left to modelgatewayclient's internal
	// default, which lives in TWO places (client.go and image.go) and which
	// nothing could previously state or override. The default below is the
	// value the library already used, so this is not a behaviour change.
	GatewayAudience string
	GatewayTenantID string
	GatewayGCID     string

	// Model is the gateway logical_model_id: the YAML-declared primary,
	// overridable per binary by env for quick ops experiments.
	Model string

	// FallbackModels is the agent-declared fallback chain. It stays
	// config-declared: the env override above replaces the primary only.
	FallbackModels []string

	// PromptVersion is the composer template version stamped onto the
	// active span by promptstamping (ADR-197 M-A).
	PromptVersion string

	// GatewayCrewKind is the cost-attribution tag sent to
	// chora-model-gateway on every InvokeRequest. qgen_question suffixes
	// its sub-agent label ("qgen_question.generation") so per-step spend
	// is separable; qgen_critic is a single agent and carries no suffix.
	// Both feed live cost dashboards, so neither is free to rename.
	GatewayCrewKind string

	// ComposeModel / ComposeFallbackModels / ComposePromptVersion are the
	// qgen_generate mode=compose composer's model (qgen_question only; the
	// YAML "compose" sub-agent, QGEN_QUESTION_COMPOSE_MODEL overrides the
	// primary).
	ComposeModel          string
	ComposeFallbackModels []string
	ComposePromptVersion  string
	// RenderBucket is the object-store bucket the renderer writes scene
	// images to (QGEN_RENDER_BUCKET, required by qgen_renderer only; the
	// kennel signs reads from it, ADR-254 D12).
	RenderBucket string
	// ImageTimeout is the renderer's per-attempt Invoke ceiling and
	// ImageRetry its bounded retry on a throttled image model.
	ImageTimeout time.Duration
	ImageRetry   modelgatewayclient.ImageRetryPolicy

	ChoraEnv string
}

// EnvOr returns os.Getenv(name) if non-empty, else fallback. POC defaults
// are coupled to the deploy-sandbox.sh PATCH path; production deploys set
// everything explicitly via Terraform-managed env per [[secrets-and-env]].
func EnvOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

// SafePrefix returns the first n characters of s, padded with "…" if
// truncated. Used to log identifiers (GCID, etc.) without printing them
// in full.
func SafePrefix(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// LoadQuestionConfig resolves the qgen_question binary's boot configuration
// from the process env plus the embedded agentconfig YAML.
//
// AGENT-DRIVEN tiering (CR qgen 2026-06-01): the per-sub-agent model tier +
// fallback chain is config-declared in the YAML (single source of truth).
// generation runs the HIGH tier (gemini-3.1-pro-preview) for question
// quality; QGEN_QUESTION_GENERATION_MODEL remains available for ops
// experiments but the YAML primary is the source of truth and the fallback
// chain is never overridable by env.
func LoadQuestionConfig() (Config, error) {
	cfg, err := loadCommon(CrewKindQuestion)
	if err != nil {
		return Config{}, err
	}

	qcfg, err := agentconfig.QGenQuestion()
	if err != nil {
		return Config{}, fmt.Errorf("qgen_question: load agent config: %w", err)
	}
	generationCfg, err := qcfg.Sub("generation")
	if err != nil {
		return Config{}, fmt.Errorf("qgen_question: %w", err)
	}

	cfg.ModelLabel = "generation"
	cfg.GatewayCrewKind = CrewKindQuestion + ".generation"
	cfg.Model = EnvOr("QGEN_QUESTION_GENERATION_MODEL", generationCfg.PrimaryModel)
	cfg.FallbackModels = generationCfg.FallbackModels
	cfg.PromptVersion = generationCfg.PromptVersion
	composeCfg, err := qcfg.Sub("compose")
	if err != nil {
		return Config{}, fmt.Errorf("qgen_question: %w", err)
	}
	cfg.ComposeModel = EnvOr("QGEN_QUESTION_COMPOSE_MODEL", composeCfg.PrimaryModel)
	cfg.ComposeFallbackModels = composeCfg.FallbackModels
	cfg.ComposePromptVersion = composeCfg.PromptVersion

	return cfg, nil
}

// LoadCriticConfig resolves the qgen_critic binary's boot configuration.
//
// AGENT-DRIVEN tiering (CR qgen 2026-06-01): the critic is the CHEAP text
// tier, config-declared in the embedded YAML (CHEAP = gemini-3.5-flash then
// gemini-2.5-flash). QGEN_CRITIC_MODEL overrides the primary only.
func LoadCriticConfig() (Config, error) {
	cfg, err := loadCommon(CrewKindCritic)
	if err != nil {
		return Config{}, err
	}

	ccfg, err := agentconfig.QGenCritic()
	if err != nil {
		return Config{}, fmt.Errorf("qgen_critic: load agent config: %w", err)
	}
	criticCfg, err := ccfg.Sub("critic")
	if err != nil {
		return Config{}, fmt.Errorf("qgen_critic: %w", err)
	}

	cfg.ModelLabel = "critic"
	cfg.GatewayCrewKind = CrewKindCritic
	cfg.Model = EnvOr("QGEN_CRITIC_MODEL", criticCfg.PrimaryModel)
	cfg.FallbackModels = criticCfg.FallbackModels
	cfg.PromptVersion = criticCfg.PromptVersion

	return cfg, nil
}

// Renderer env (binary-specific, per feedback_no_inline_config: names only,
// values from the container env).
const (
	envRenderBucket        = "QGEN_RENDER_BUCKET"
	envRendererModel       = "QGEN_RENDERER_MODEL"
	envRenderTimeoutSecs   = "QGEN_RENDER_TIMEOUT_SECONDS"
	envRenderRetryAttempts = "QGEN_RENDER_RETRY_ATTEMPTS"
	envRenderRetryMaxDelay = "QGEN_RENDER_RETRY_MAX_DELAY_SECONDS"
)

// LoadRendererConfig resolves the qgen_renderer binary's boot configuration:
// the image model from the embedded YAML (primary gemini-3-pro-image, the
// Pro image model the owner chose for quality, CHO-2220), the render bucket
// (required: without it there is nowhere to put the image and nothing to
// return), the per-attempt timeout and the retry budget for a throttled model.
func LoadRendererConfig() (Config, error) {
	cfg, err := loadCommon(CrewKindRenderer)
	if err != nil {
		return Config{}, err
	}
	rcfg, err := agentconfig.QGenRenderer()
	if err != nil {
		return Config{}, fmt.Errorf("qgen_renderer: load agent config: %w", err)
	}
	rendererCfg, err := rcfg.Sub("renderer")
	if err != nil {
		return Config{}, fmt.Errorf("qgen_renderer: %w", err)
	}
	bucket := os.Getenv(envRenderBucket)
	if bucket == "" {
		return Config{}, fmt.Errorf("qgen_renderer: %s required (the object-store bucket scene images are written to; the kennel signs reads from it, ADR-254 D12)", envRenderBucket)
	}
	cfg.ModelLabel = "renderer"
	cfg.GatewayCrewKind = CrewSurface
	cfg.Model = EnvOr(envRendererModel, rendererCfg.PrimaryModel)
	cfg.FallbackModels = rendererCfg.FallbackModels
	cfg.PromptVersion = rendererCfg.PromptVersion
	cfg.RenderBucket = bucket
	timeoutSecs, err := envFloat(envRenderTimeoutSecs, 120)
	if err != nil {
		return Config{}, err
	}
	cfg.ImageTimeout = time.Duration(timeoutSecs * float64(time.Second))
	attempts, err := envFloat(envRenderRetryAttempts, 4)
	if err != nil {
		return Config{}, err
	}
	maxDelay, err := envFloat(envRenderRetryMaxDelay, 30)
	if err != nil {
		return Config{}, err
	}
	cfg.ImageRetry = modelgatewayclient.ImageRetryPolicy{Attempts: int(attempts), BaseDelay: 2 * time.Second, MaxDelay: time.Duration(maxDelay * float64(time.Second))}
	return cfg, nil
}

func envFloat(name string, fallback float64) (float64, error) {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("%s must be a positive number, got %q", name, raw)
	}
	return v, nil
}

// loadCommon resolves everything the three binaries share: project, the ADK
// session app name and the ADR-163 gateway identity. The fail-loud guards
// live here because every binary carries them verbatim.
func loadCommon(binary string) (Config, error) {
	projectID := os.Getenv("CHORA_PROJECT_ID")
	if projectID == "" {
		return Config{}, errors.New("CHORA_PROJECT_ID required")
	}

	// Phase 3.1, chora-model-gateway wiring. Endpoint defaults to the
	// canonical public ingress; tenant_id + gcid are required (no
	// silent-stub fallback per feedback_no_stubs_real_wiring) and
	// surfaced via the container env.
	gatewayTenantID := os.Getenv("CHORA_GATEWAY_TENANT_ID")
	gatewayGCID := os.Getenv("CHORA_GATEWAY_GCID")
	if gatewayTenantID == "" || gatewayGCID == "" {
		return Config{}, fmt.Errorf("%s: CHORA_GATEWAY_TENANT_ID + CHORA_GATEWAY_GCID required "+
			"(Phase 3.1 cutover per ADR-163; no in-memory fallback per feedback_no_stubs_real_wiring)", binary)
	}

	return Config{
		Binary:    binary,
		ProjectID: projectID,
		// ADK session AppName label (optional; the dispatch lane keys off the
		// binary name regardless).
		AgentAppName:    os.Getenv("CHORA_AGENT_APP_NAME"),
		GatewayEndpoint: EnvOr("CHORA_GATEWAY_ENDPOINT", "gateway.chora.site:443"),
		GatewayAudience: EnvOr("CHORA_GATEWAY_AUDIENCE", "https://gateway.chora.site"),
		GatewayTenantID: gatewayTenantID,
		GatewayGCID:     gatewayGCID,
		ChoraEnv:        EnvOr("CHORA_ENV", "dev"),
	}, nil
}

// LogAttrs returns the boot log's slog key/value pairs in a fixed order.
// The model keys are labelled per sub-agent ("generation_model" for
// qgen_question, "critic_model" for qgen_critic) and creation_endpoint is
// emitted only by the binary that resolves one.
//
// The GCID is truncated to an 8-character prefix. It is never logged whole.
func (c Config) LogAttrs() []any {
	attrs := []any{
		"project", c.ProjectID,
		"agent_app_name", c.AgentAppName,
		c.ModelLabel + "_model", c.Model,
		c.ModelLabel + "_fallback", c.FallbackModels,
		"gateway_endpoint", c.GatewayEndpoint,
		"gateway_tenant_id", c.GatewayTenantID,
		"gateway_gcid_prefix", SafePrefix(c.GatewayGCID, 8),
		"surface", CrewSurface,
	}
	if c.ComposeModel != "" {
		attrs = append(attrs, "compose_model", c.ComposeModel, "compose_fallback", c.ComposeFallbackModels)
	}
	if c.RenderBucket != "" {
		attrs = append(attrs, "render_bucket", c.RenderBucket, "image_timeout", c.ImageTimeout.String(), "image_retry_attempts", c.ImageRetry.Attempts)
	}
	return append(attrs, "chora_env", c.ChoraEnv)
}

// GatewayConfig is the chora-model-gateway client wiring for this binary
// (ADR-163 Phase 3.1). Every LLM call routes through the gateway at
// GatewayEndpoint; the gateway's per-agent routing policy YAML picks the
// actual region, which is why the legacy per-agent *_LOCATION env vars are
// ignored.
//
// The declared fallback chain travels with the request: the gateway HONOURS
// it rather than hardcoding a per-tier ladder (AGENT-DRIVEN tiering, CR qgen
// 2026-06-01).
func (c Config) GatewayConfig() modelgatewayclient.Config {
	return modelgatewayclient.Config{
		Endpoint:         c.GatewayEndpoint,
		LogicalModelID:   c.Model,
		FallbackModelIDs: c.FallbackModels,
		AgentID:          c.Binary,
		CrewKind:         c.GatewayCrewKind,
		Surface:          CrewSurface,
		TenantID:         c.GatewayTenantID,
		GCID:             c.GatewayGCID,
		Audience:         c.GatewayAudience,
	}
}

// ComposeGatewayConfig is the gateway client of the mode=compose composer:
// the same crew identity and surface, the composer's own model and chain,
// cost-attributed as qgen_question.compose.
func (c Config) ComposeGatewayConfig() modelgatewayclient.Config {
	gw := c.GatewayConfig()
	gw.LogicalModelID = c.ComposeModel
	gw.FallbackModelIDs = c.ComposeFallbackModels
	gw.CrewKind = CrewKindQuestion + ".compose"
	return gw
}
