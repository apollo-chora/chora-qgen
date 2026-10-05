package boot

import (
	"context"
	"fmt"
	"log/slog"

	"google.golang.org/adk/agent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"

	"github.com/apollo-chora/chora-adk-common/agentdispatch"
	"github.com/apollo-chora/chora-adk-common/modelgatewayclient"
	"github.com/apollo-chora/chora-adk-common/tracing"
)

// Process-level collaborators, as package variables so a unit test can prove
// the boot composition (which identities reach the subscriber) without ADC, a
// network dial or a broker. Production never reassigns them.
var (
	initTracing   = tracing.Init
	newGatewayLLM = func(ctx context.Context, cfg Config) (adkmodel.LLM, error) {
		return modelgatewayclient.New(ctx, cfg.GatewayConfig())
	}
	newComposeLLM = func(ctx context.Context, cfg Config) (adkmodel.LLM, error) {
		return modelgatewayclient.New(ctx, cfg.ComposeGatewayConfig())
	}
	newImageInvoker = func(ctx context.Context, cfg Config) (ImageInvoker, error) {
		gw := cfg.GatewayConfig()
		gw.CallTimeout = cfg.ImageTimeout
		return modelgatewayclient.NewImageInvoker(ctx, gw, cfg.ImageRetry)
	}
	newObjectStore  = func(ctx context.Context, bucket string) (ObjectStore, error) { return NewS3ObjectStore(ctx, bucket) }
	serveSubscriber = agentdispatch.RunSubscriberOnly
)

// RunQuestion boots and serves qgen_question subscriber-only on role
// qgen_generate (ADR-254 D6). It returns rather than exits; a non-nil error
// means the pod must die.
func RunQuestion(ctx context.Context, args []string) error {
	if err := refuseArgs(CrewKindQuestion, args); err != nil {
		return err
	}
	cfg, err := LoadQuestionConfig()
	if err != nil {
		return err
	}
	return serveLLM(ctx, cfg, NewQuestionRoot, NewQuestionPlugins)
}

// RunCritic boots and serves qgen_critic subscriber-only on role qgen_critique.
func RunCritic(ctx context.Context, args []string) error {
	if err := refuseArgs(CrewKindCritic, args); err != nil {
		return err
	}
	cfg, err := LoadCriticConfig()
	if err != nil {
		return err
	}
	return serveLLM(ctx, cfg, func(c Config, llm, _ adkmodel.LLM) (agent.Agent, error) { return NewCriticAgent(c, llm) }, NewCriticPlugins)
}

// RunRenderer boots and serves qgen_renderer subscriber-only on role
// qgen_render: the gateway image invoker + the render bucket instead of an
// ADK LLM.
func RunRenderer(ctx context.Context, args []string) error {
	if err := refuseArgs(CrewKindRenderer, args); err != nil {
		return err
	}
	cfg, err := LoadRendererConfig()
	if err != nil {
		return err
	}
	return withTracing(ctx, cfg, func(ctx context.Context) error {
		invoker, err := newImageInvoker(ctx, cfg)
		if err != nil {
			return fmt.Errorf("modelgatewayclient.NewImageInvoker(%s @ %s): %w", cfg.Model, cfg.GatewayEndpoint, err)
		}
		store, err := newObjectStore(ctx, cfg.RenderBucket)
		if err != nil {
			return fmt.Errorf("render bucket %s: %w", cfg.RenderBucket, err)
		}
		root, err := NewRendererAgent(cfg, invoker, store)
		if err != nil {
			return err
		}
		plugins, err := NewRendererPlugins()
		if err != nil {
			return err
		}
		serveCfg, err := dispatchServeConfig(cfg, root, plugins)
		if err != nil {
			return err
		}
		return serveSubscriber(ctx, serveCfg, agentdispatch.RunOptions{})
	})
}

// refuseArgs rejects any command-line argument: the ADK web launcher and its
// "web -port ..." command line are gone (ADR-254 D6); a container whose
// command was not updated dies with the cause in its exit line.
func refuseArgs(binary string, args []string) error {
	if len(args) == 0 {
		return nil
	}
	return fmt.Errorf("%s takes no arguments, got %q: the ADK web launcher "+
		"(\"web -port ...\") was removed by ADR-254 D6 and this binary "+
		"is subscriber-only; update the container command", binary, args)
}

// withTracing wires the OTel exporter + W3C propagator BEFORE any
// agent / runner / plugin construction, logs the boot line and runs body.
func withTracing(ctx context.Context, cfg Config, body func(context.Context) error) error {
	traceShutdown, err := initTracing(ctx, cfg.Binary)
	if err != nil {
		return fmt.Errorf("tracing.Init: %w", err)
	}
	defer func() {
		if err := traceShutdown(context.Background()); err != nil {
			slog.Error("trace shutdown error", "err", err)
		}
	}()
	slog.Info(cfg.Binary+" boot", cfg.LogAttrs()...)
	return body(ctx)
}

// serveLLM is the shared boot of the two LLM binaries: the gateway-fronted
// LLM (ADR-163), the root agent, the plugin chain, the subscriber. Model
// selection is AGENT-DRIVEN (the YAML primary + declared fallback chain); no
// agent-side mana gate, the gateway meters (ADR-177 / ADR-254 A3).
func serveLLM(
	ctx context.Context, cfg Config,
	newRoot func(Config, adkmodel.LLM, adkmodel.LLM) (agent.Agent, error),
	newPlugins func() ([]*plugin.Plugin, error),
) error {
	return withTracing(ctx, cfg, func(ctx context.Context) error {
		llm, err := newGatewayLLM(ctx, cfg)
		if err != nil {
			return fmt.Errorf("modelgatewayclient.New(%s, %s @ %s): %w", cfg.ModelLabel, cfg.Model, cfg.GatewayEndpoint, err)
		}
		// The composer's own gateway client (qgen_question only): built when
		// the YAML declares a compose model, so mode=compose is served by a
		// real client or refused by name, never faked.
		var composeLLM adkmodel.LLM
		if cfg.ComposeModel != "" {
			composeLLM, err = newComposeLLM(ctx, cfg)
			if err != nil {
				return fmt.Errorf("modelgatewayclient.New(compose, %s @ %s): %w", cfg.ComposeModel, cfg.GatewayEndpoint, err)
			}
		}
		root, err := newRoot(cfg, llm, composeLLM)
		if err != nil {
			return err
		}
		plugins, err := newPlugins()
		if err != nil {
			return err
		}
		serveCfg, err := dispatchServeConfig(cfg, root, plugins)
		if err != nil {
			return err
		}
		return serveSubscriber(ctx, serveCfg, agentdispatch.RunOptions{})
	})
}

// dispatchServeConfig derives the lane identity from the binary; an unknown
// binary is refused rather than guessed. Every qgen role runs stateless
// per-dispatch sessions.
func dispatchServeConfig(cfg Config, root agent.Agent, plugins []*plugin.Plugin) (agentdispatch.ServeConfig, error) {
	role := DispatchRole(cfg.Binary)
	if role == "" {
		return agentdispatch.ServeConfig{}, fmt.Errorf("binary %q has no dispatch role: refusing to guess one", cfg.Binary)
	}
	return agentdispatch.ServeConfig{
		AgentRole:   role,
		ServiceName: ServiceName(cfg.Binary),
		AppName:     cfg.AgentAppName,
		RootAgent:   root,
		Sessions:    session.InMemoryService(),
		Plugins:     runner.PluginConfig{Plugins: plugins},
	}, nil
}
