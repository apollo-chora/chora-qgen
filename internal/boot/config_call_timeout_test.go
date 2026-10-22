package boot

import (
	"testing"
	"time"
)

func TestCallTimeout_defaultsToGatewayClientCeiling_LoadQuestionConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	t.Setenv(envModelCallTimeoutSecs, "")
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}
	if cfg.CallTimeout != 90*time.Second {
		t.Fatalf("CallTimeout = %s, want the 90s default", cfg.CallTimeout)
	}
	if got := cfg.GatewayConfig().CallTimeout; got != cfg.CallTimeout {
		t.Fatalf("GatewayConfig().CallTimeout = %s, want %s", got, cfg.CallTimeout)
	}
}

func TestCallTimeout_honoursEnv_LoadQuestionConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	t.Setenv(envModelCallTimeoutSecs, "240")
	cfg, err := LoadQuestionConfig()
	if err != nil {
		t.Fatalf("LoadQuestionConfig: %v", err)
	}
	if cfg.CallTimeout != 240*time.Second {
		t.Fatalf("CallTimeout = %s, want 4m0s", cfg.CallTimeout)
	}
	if got := cfg.GatewayConfig().CallTimeout; got != 240*time.Second {
		t.Fatalf("GatewayConfig().CallTimeout = %s, want 4m0s", got)
	}
	if got := cfg.ComposeGatewayConfig().CallTimeout; got != 240*time.Second {
		t.Fatalf("ComposeGatewayConfig().CallTimeout = %s, want 4m0s", got)
	}
}

func TestCallTimeout_rejectsNonNumeric_LoadQuestionConfig(t *testing.T) {
	setBootEnv(t, minimalQuestionEnv())
	t.Setenv(envModelCallTimeoutSecs, "not-a-number")
	if _, err := LoadQuestionConfig(); err == nil {
		t.Fatal("LoadQuestionConfig accepted a non-numeric call timeout, want an error")
	}
}
