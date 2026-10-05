// RED-first for D6 step 1 (owner ruling R21, sequenced).
//
// CHORA_GATEWAY_AUDIENCE must be READ HERE and defaulted EXPLICITLY, so the
// audience becomes a stateable value instead of one the shared library invents.
// Today `modelgatewayclient` defaults it in TWO places, client.go:181-182 and
// image.go:110-111, and no caller anywhere sets it.
//
// ⚠ This test asserts the value is CARRIED, not that the library stops
// defaulting. Removing those two defaults is step 4 and is gated on the
// manifests reaching the live cluster first: with 12 constructor sites across 9
// agents and no CHORA_GATEWAY_AUDIENCE in any manifest, a removal today fails
// every agent at construction.
package boot

import "testing"

func TestGatewayAudience_defaultsExplicitly_LoadQuestionConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}
	if cfg.GatewayAudience != "https://gateway.chora.site" {
		t.Fatalf("GatewayAudience = %q, want the explicit default %q",
			cfg.GatewayAudience, "https://gateway.chora.site")
	}
}

func TestGatewayAudience_honoursEnv_LoadQuestionConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	t.Setenv("CHORA_GATEWAY_AUDIENCE", "https://gateway.example.invalid")
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}
	if cfg.GatewayAudience != "https://gateway.example.invalid" {
		t.Fatalf("GatewayAudience = %q, want the env value", cfg.GatewayAudience)
	}
}

func TestGatewayAudience_defaultsExplicitly_LoadCriticConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	cfg, err := LoadCriticConfig()
	if err != nil {
		t.Fatalf("LoadCriticConfig: %v", err)
	}
	if cfg.GatewayAudience != "https://gateway.chora.site" {
		t.Fatalf("GatewayAudience = %q, want the explicit default %q",
			cfg.GatewayAudience, "https://gateway.chora.site")
	}
}

func TestGatewayAudience_honoursEnv_LoadCriticConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	t.Setenv("CHORA_GATEWAY_AUDIENCE", "https://gateway.example.invalid")
	cfg, err := LoadCriticConfig()
	if err != nil {
		t.Fatalf("LoadCriticConfig: %v", err)
	}
	if cfg.GatewayAudience != "https://gateway.example.invalid" {
		t.Fatalf("GatewayAudience = %q, want the env value", cfg.GatewayAudience)
	}
}

func TestGatewayAudience_defaultsExplicitly_LoadRendererConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	// ⚠ set directly: QGEN_RENDER_BUCKET is not in bootEnvKeys, so passing
	// it through setBootEnv's override map is silently ignored.
	t.Setenv("QGEN_RENDER_BUCKET", "render-bucket")
	cfg, err := LoadRendererConfig()
	if err != nil {
		t.Fatalf("LoadRendererConfig: %v", err)
	}
	if cfg.GatewayAudience != "https://gateway.chora.site" {
		t.Fatalf("GatewayAudience = %q, want the explicit default %q",
			cfg.GatewayAudience, "https://gateway.chora.site")
	}
}

func TestGatewayAudience_honoursEnv_LoadRendererConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	t.Setenv("QGEN_RENDER_BUCKET", "render-bucket")
	t.Setenv("CHORA_GATEWAY_AUDIENCE", "https://gateway.example.invalid")
	cfg, err := LoadRendererConfig()
	if err != nil {
		t.Fatalf("LoadRendererConfig: %v", err)
	}
	if cfg.GatewayAudience != "https://gateway.example.invalid" {
		t.Fatalf("GatewayAudience = %q, want the env value", cfg.GatewayAudience)
	}
}
