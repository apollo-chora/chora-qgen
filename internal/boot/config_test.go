package boot

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

// bootEnvKeys is every environment variable the two Load*Config functions
// read. Tests set ALL of them explicitly (empty string when the case wants
// them unset) so a developer machine that already exports CHORA_PROJECT_ID
// can never leak into an assertion.
var bootEnvKeys = []string{
	"CHORA_PROJECT_ID",
	"CHORA_AGENT_APP_NAME",
	"QGEN_QUESTION_GENERATION_MODEL",
	"QGEN_CRITIC_MODEL",
	"CHORA_GATEWAY_ENDPOINT",
	"CHORA_GATEWAY_TENANT_ID",
	"CHORA_GATEWAY_GCID",
	"CHORA_ENV",
}

// setBootEnv pins every boot env var for the duration of one test. Keys
// absent from overrides are pinned to "" (the "unset" case). t.Setenv
// restores the pre-test value on cleanup. CREATION_GRPC_ENDPOINT stays in
// the pinned list so a stray shell value cannot leak into a test.
func setBootEnv(t *testing.T, overrides map[string]string) {
	t.Helper()
	for _, k := range bootEnvKeys {
		t.Setenv(k, overrides[k])
	}
}

// minimalQuestionEnv is the smallest env that loads without error: project
// plus the two ADR-163 gateway identity vars.
func minimalQuestionEnv() map[string]string {
	return map[string]string{
		"CHORA_PROJECT_ID":        "chora-489812",
		"CHORA_GATEWAY_TENANT_ID": "tenant-abc",
		"CHORA_GATEWAY_GCID":      "0192f3a4-b5c6-7d8e-9f01-234567890abc",
	}
}

// ---------------------------------------------------------------------------
// EnvOr
// ---------------------------------------------------------------------------

func TestEnvOr(t *testing.T) {
	t.Setenv("CHORA_BOOT_TEST_SET", "from-env")
	t.Setenv("CHORA_BOOT_TEST_EMPTY", "")
	if err := os.Unsetenv("CHORA_BOOT_TEST_ABSENT"); err != nil {
		t.Fatalf("unsetenv: %v", err)
	}

	cases := []struct {
		name     string
		key      string
		fallback string
		want     string
	}{
		{"set wins over fallback", "CHORA_BOOT_TEST_SET", "fb", "from-env"},
		{"empty falls back", "CHORA_BOOT_TEST_EMPTY", "fb", "fb"},
		{"absent falls back", "CHORA_BOOT_TEST_ABSENT", "fb", "fb"},
		{"absent with empty fallback", "CHORA_BOOT_TEST_ABSENT", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := EnvOr(tc.key, tc.fallback); got != tc.want {
				t.Errorf("EnvOr(%q, %q) = %q, want %q", tc.key, tc.fallback, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// SafePrefix, including the truncation boundary
// ---------------------------------------------------------------------------

func TestSafePrefix(t *testing.T) {
	cases := []struct {
		name string
		in   string
		n    int
		want string
	}{
		{"shorter than n is returned whole", "abc", 8, "abc"},
		{"exactly n is returned whole, no ellipsis", "abcdefgh", 8, "abcdefgh"},
		{"one over n truncates", "abcdefghi", 8, "abcdefgh…"},
		{"empty stays empty", "", 8, ""},
		{"zero n on empty stays empty", "", 0, ""},
		{"zero n truncates everything", "abc", 0, "…"},
		{"gcid is cut to 8 chars", "0192f3a4-b5c6-7d8e-9f01-234567890abc", 8, "0192f3a4…"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SafePrefix(tc.in, tc.n); got != tc.want {
				t.Errorf("SafePrefix(%q, %d) = %q, want %q", tc.in, tc.n, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// LoadQuestionConfig
// ---------------------------------------------------------------------------

func TestLoadQuestionConfig_defaults(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())

	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}

	assertField(t, "Binary", cfg.Binary, "qgen_question")
	assertField(t, "ModelLabel", cfg.ModelLabel, "generation")
	assertField(t, "ProjectID", cfg.ProjectID, "chora-489812")
	assertField(t, "AgentAppName", cfg.AgentAppName, "")
	assertField(t, "GatewayEndpoint", cfg.GatewayEndpoint, "gateway.chora.site:443")
	assertField(t, "GatewayTenantID", cfg.GatewayTenantID, "tenant-abc")
	assertField(t, "GatewayGCID", cfg.GatewayGCID, "0192f3a4-b5c6-7d8e-9f01-234567890abc")
	// From the embedded agentconfig YAML: generation is the HIGH tier.
	assertField(t, "Model", cfg.Model, "gemini-3.1-pro-preview")
	assertField(t, "PromptVersion", cfg.PromptVersion, "v1")
	assertField(t, "ChoraEnv", cfg.ChoraEnv, "dev")

	if len(cfg.FallbackModels) != 1 || cfg.FallbackModels[0] != "gemini-2.5-pro" {
		t.Errorf("FallbackModels = %v, want [gemini-2.5-pro]", cfg.FallbackModels)
	}
}

func TestLoadQuestionConfig_everyOverrideApplied(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_AGENT_APP_NAME"] = "qgen-question-iter1"
	env["QGEN_QUESTION_GENERATION_MODEL"] = "gemini-ops-experiment"
	env["CHORA_GATEWAY_ENDPOINT"] = "localhost:9090"
	env["CHORA_ENV"] = "prod"
	setBootEnv(t, env)

	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}

	assertField(t, "AgentAppName", cfg.AgentAppName, "qgen-question-iter1")
	assertField(t, "Model", cfg.Model, "gemini-ops-experiment")
	assertField(t, "GatewayEndpoint", cfg.GatewayEndpoint, "localhost:9090")
	assertField(t, "ChoraEnv", cfg.ChoraEnv, "prod")

	// The env override replaces the primary ONLY. Tier and fallback chain
	// stay config-declared per the agentconfig YAML contract.
	if len(cfg.FallbackModels) != 1 || cfg.FallbackModels[0] != "gemini-2.5-pro" {
		t.Errorf("env model override clobbered the config-declared fallback chain: %v", cfg.FallbackModels)
	}
}

func TestLoadQuestionConfig_missingProjectFailsLoud(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_PROJECT_ID"] = ""
	setBootEnv(t, env)

	_, err := LoadQuestionConfig()
	if err == nil {
		t.Fatal("LoadQuestionConfig accepted an empty CHORA_PROJECT_ID")
	}
	if !strings.Contains(err.Error(), "CHORA_PROJECT_ID") {
		t.Errorf("error does not name the missing var: %v", err)
	}
}

func TestLoadQuestionConfig_missingGatewayTenantFailsLoud(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_TENANT_ID"] = ""
	setBootEnv(t, env)

	_, err := LoadQuestionConfig()
	if err == nil {
		t.Fatal("LoadQuestionConfig accepted an empty CHORA_GATEWAY_TENANT_ID")
	}
	assertGatewayIdentityError(t, err)
}

func TestLoadQuestionConfig_missingGatewayGCIDFailsLoud(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_GCID"] = ""
	setBootEnv(t, env)

	_, err := LoadQuestionConfig()
	if err == nil {
		t.Fatal("LoadQuestionConfig accepted an empty CHORA_GATEWAY_GCID")
	}
	assertGatewayIdentityError(t, err)
}

// ---------------------------------------------------------------------------
// LoadCriticConfig
// ---------------------------------------------------------------------------

func TestLoadCriticConfig_defaults(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())

	cfg, err := LoadCriticConfig()
	if err != nil {
		t.Fatalf("LoadCriticConfig: %v", err)
	}

	assertField(t, "Binary", cfg.Binary, "qgen_critic")
	assertField(t, "ModelLabel", cfg.ModelLabel, "critic")
	assertField(t, "ProjectID", cfg.ProjectID, "chora-489812")
	assertField(t, "AgentAppName", cfg.AgentAppName, "")
	assertField(t, "GatewayEndpoint", cfg.GatewayEndpoint, "gateway.chora.site:443")
	// From the embedded agentconfig YAML: the critic is the CHEAP tier.
	assertField(t, "Model", cfg.Model, "gemini-3.5-flash")
	assertField(t, "PromptVersion", cfg.PromptVersion, "v1")
	assertField(t, "ChoraEnv", cfg.ChoraEnv, "dev")

	if len(cfg.FallbackModels) != 1 || cfg.FallbackModels[0] != "gemini-2.5-flash" {
		t.Errorf("FallbackModels = %v, want [gemini-2.5-flash]", cfg.FallbackModels)
	}
}

func TestLoadCriticConfig_modelOverride(t *testing.T) {
	env := minimalQuestionEnv()
	env["QGEN_CRITIC_MODEL"] = "gemini-critic-experiment"
	// The question binary's override key must NOT bleed into the critic.
	env["QGEN_QUESTION_GENERATION_MODEL"] = "gemini-wrong-binary"
	setBootEnv(t, env)

	cfg, err := LoadCriticConfig()
	if err != nil {
		t.Fatalf("LoadCriticConfig: %v", err)
	}
	assertField(t, "Model", cfg.Model, "gemini-critic-experiment")
}

func TestLoadCriticConfig_missingProjectFailsLoud(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_PROJECT_ID"] = ""
	setBootEnv(t, env)

	_, err := LoadCriticConfig()
	if err == nil {
		t.Fatal("LoadCriticConfig accepted an empty CHORA_PROJECT_ID")
	}
	if !strings.Contains(err.Error(), "CHORA_PROJECT_ID") {
		t.Errorf("error does not name the missing var: %v", err)
	}
}

func TestLoadCriticConfig_missingGatewayTenantFailsLoud(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_TENANT_ID"] = ""
	setBootEnv(t, env)

	_, err := LoadCriticConfig()
	if err == nil {
		t.Fatal("LoadCriticConfig accepted an empty CHORA_GATEWAY_TENANT_ID")
	}
	assertGatewayIdentityError(t, err)
}

func TestLoadCriticConfig_missingGatewayGCIDFailsLoud(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_GCID"] = ""
	setBootEnv(t, env)

	_, err := LoadCriticConfig()
	if err == nil {
		t.Fatal("LoadCriticConfig accepted an empty CHORA_GATEWAY_GCID")
	}
	assertGatewayIdentityError(t, err)
}

// assertGatewayIdentityError pins the ADR-163 rationale onto the fail-loud
// message. The message is the only thing an operator sees when a deploy
// forgets the gateway identity vars, so its content is load-bearing.
func assertGatewayIdentityError(t *testing.T, err error) {
	t.Helper()
	for _, want := range []string{
		"CHORA_GATEWAY_TENANT_ID",
		"CHORA_GATEWAY_GCID",
		"ADR-163",
		"feedback_no_stubs_real_wiring",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("gateway identity error missing %q: %v", want, err)
		}
	}
}

// ---------------------------------------------------------------------------
// LogAttrs
// ---------------------------------------------------------------------------

func TestConfig_LogAttrs_questionShapeAndOrder(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}

	attrs := cfg.LogAttrs()
	if len(attrs)%2 != 0 {
		t.Fatalf("LogAttrs returned an odd number of items (%d); slog needs key/value pairs", len(attrs))
	}

	wantKeys := []string{
		"project",
		"agent_app_name",
		"generation_model",
		"generation_fallback",
		"gateway_endpoint",
		"gateway_tenant_id",
		"gateway_gcid_prefix",
		"surface",
		"compose_model",
		"compose_fallback",
		"chora_env",
	}
	assertLogKeys(t, attrs, wantKeys)

	got := logAttrMap(t, attrs)
	assertField(t, "project", got["project"], "chora-489812")
	assertField(t, "generation_model", got["generation_model"], "gemini-3.1-pro-preview")
}

func TestConfig_LogAttrs_criticShapeAndOrder(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadCriticConfig()
	if err != nil {
		t.Fatalf("LoadCriticConfig: %v", err)
	}

	// Same shape as the question binary MINUS creation_endpoint, and with
	// the model keys labelled for the critic sub-agent.
	assertLogKeys(t, cfg.LogAttrs(), []string{
		"project",
		"agent_app_name",
		"critic_model",
		"critic_fallback",
		"gateway_endpoint",
		"gateway_tenant_id",
		"gateway_gcid_prefix",
		"surface",
		"chora_env",
	})
}

func TestConfig_LogAttrs_neverLogsTheFullGCID(t *testing.T) {
	full := "0192f3a4-b5c6-7d8e-9f01-234567890abc"
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_GCID"] = full
	setBootEnv(t, env)

	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}

	got := logAttrMap(t, cfg.LogAttrs())
	if got["gateway_gcid_prefix"] != "0192f3a4…" {
		t.Errorf("gateway_gcid_prefix = %v, want the 8-char truncation", got["gateway_gcid_prefix"])
	}
	for k, v := range got {
		if s, ok := v.(string); ok && strings.Contains(s, full) {
			t.Errorf("log attr %q leaks the full GCID: %q", k, s)
		}
	}
}

func assertLogKeys(t *testing.T, attrs []any, want []string) {
	t.Helper()
	var got []string
	for i := 0; i < len(attrs); i += 2 {
		k, ok := attrs[i].(string)
		if !ok {
			t.Fatalf("LogAttrs item %d is not a string key: %T", i, attrs[i])
		}
		got = append(got, k)
	}
	if len(got) != len(want) {
		t.Fatalf("LogAttrs keys = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("LogAttrs key %d = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

func logAttrMap(t *testing.T, attrs []any) map[string]any {
	t.Helper()
	m := make(map[string]any, len(attrs)/2)
	for i := 0; i+1 < len(attrs); i += 2 {
		k, ok := attrs[i].(string)
		if !ok {
			t.Fatalf("LogAttrs item %d is not a string key: %T", i, attrs[i])
		}
		m[k] = attrs[i+1]
	}
	return m
}

// ---------------------------------------------------------------------------
// GatewayConfig: the chora-model-gateway wiring, ADR-163 Phase 3.1
// ---------------------------------------------------------------------------

func TestConfig_GatewayConfig_question(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}

	gw := cfg.GatewayConfig()
	assertField(t, "Endpoint", gw.Endpoint, "gateway.chora.site:443")
	assertField(t, "LogicalModelID", gw.LogicalModelID, "gemini-3.1-pro-preview")
	// AgentID is the registry name the gateway's routing policy keys on.
	assertField(t, "AgentID", gw.AgentID, "qgen_question")
	// CrewKind is the cost-attribution tag. The question binary tags the
	// sub-agent so per-step spend is separable.
	assertField(t, "CrewKind", gw.CrewKind, "qgen_question.generation")
	assertField(t, "TenantID", gw.TenantID, "tenant-abc")
	assertField(t, "GCID", gw.GCID, "0192f3a4-b5c6-7d8e-9f01-234567890abc")
	if len(gw.FallbackModelIDs) != 1 || gw.FallbackModelIDs[0] != "gemini-2.5-pro" {
		t.Errorf("FallbackModelIDs = %v, want [gemini-2.5-pro]", gw.FallbackModelIDs)
	}
}

func TestConfig_GatewayConfig_critic(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadCriticConfig()
	if err != nil {
		t.Fatalf("LoadCriticConfig: %v", err)
	}

	gw := cfg.GatewayConfig()
	assertField(t, "LogicalModelID", gw.LogicalModelID, "gemini-3.5-flash")
	assertField(t, "AgentID", gw.AgentID, "qgen_critic")
	// The critic is a single agent, so its cost tag carries no sub-agent
	// suffix. Renaming this splits the cost dashboards.
	assertField(t, "CrewKind", gw.CrewKind, "qgen_critic")
	if len(gw.FallbackModelIDs) != 1 || gw.FallbackModelIDs[0] != "gemini-2.5-flash" {
		t.Errorf("FallbackModelIDs = %v, want [gemini-2.5-flash]", gw.FallbackModelIDs)
	}
}

// ---------------------------------------------------------------------------
// assertField keeps the table-style config assertions readable.
func assertField(t *testing.T, name string, got, want any) {
	t.Helper()
	if got != want {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

func TestBootPackage_carriesNoInlineStubEndpoint(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	scanned := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		body, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		scanned++
		if bytes.Contains(body, []byte("stub://")) {
			t.Errorf("%s carries an inline stub:// endpoint default", name)
		}
	}
	if scanned == 0 {
		t.Fatal("scanned no source files; the check could not have failed")
	}
}
