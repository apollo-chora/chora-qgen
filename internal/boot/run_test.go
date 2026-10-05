package boot

import (
	"context"
	"strings"
	"testing"
)

// The rest of RunQuestion / RunCritic dials the outside world (OTel, the
// gateway, then a blocking subscriber), so only
// the boot-failure path is unit-reachable. It is also the path that matters
// most: main() must receive a descriptive error to fatal on rather than a
// nil return that would leave a crew silently unserved.

func TestRunQuestion_returnsTheConfigErrorWithoutDiallingAnything(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_PROJECT_ID"] = ""
	setBootEnv(t, env)

	err := RunQuestion(context.Background(), nil)
	if err == nil {
		t.Fatal("RunQuestion returned nil with no CHORA_PROJECT_ID; main() would exit 0 on a failed boot")
	}
	if !strings.Contains(err.Error(), "CHORA_PROJECT_ID") {
		t.Errorf("error does not name the missing var: %v", err)
	}
}

func TestRunQuestion_returnsTheGatewayIdentityError(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_GCID"] = ""
	setBootEnv(t, env)

	err := RunQuestion(context.Background(), nil)
	if err == nil {
		t.Fatal("RunQuestion returned nil with no CHORA_GATEWAY_GCID")
	}
	assertGatewayIdentityError(t, err)
}

func TestRunCritic_returnsTheConfigErrorWithoutDiallingAnything(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_PROJECT_ID"] = ""
	setBootEnv(t, env)

	err := RunCritic(context.Background(), nil)
	if err == nil {
		t.Fatal("RunCritic returned nil with no CHORA_PROJECT_ID; main() would exit 0 on a failed boot")
	}
	if !strings.Contains(err.Error(), "CHORA_PROJECT_ID") {
		t.Errorf("error does not name the missing var: %v", err)
	}
}

func TestRunCritic_returnsTheGatewayIdentityError(t *testing.T) {
	env := minimalQuestionEnv()
	env["CHORA_GATEWAY_TENANT_ID"] = ""
	setBootEnv(t, env)

	err := RunCritic(context.Background(), nil)
	if err == nil {
		t.Fatal("RunCritic returned nil with no CHORA_GATEWAY_TENANT_ID")
	}
	assertGatewayIdentityError(t, err)
}
