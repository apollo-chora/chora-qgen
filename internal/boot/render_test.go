package boot

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/apollo-chora/chora-adk-common/agentdispatch"
	"github.com/apollo-chora/chora-adk-common/modelgatewayclient"
)

type fakeInvoker struct {
	reqs   []modelgatewayclient.ImageRequest
	result modelgatewayclient.ImageResult
	err    error
}

func (f *fakeInvoker) Invoke(_ context.Context, req modelgatewayclient.ImageRequest) (modelgatewayclient.ImageResult, error) {
	f.reqs = append(f.reqs, req)
	return f.result, f.err
}

type fakeStore struct {
	written  map[string][]byte
	ctypes   map[string]string
	source   []byte
	srcMime  string
	readErr  error
	writeErr error
	reads    []string
}

func (f *fakeStore) Write(_ context.Context, key string, data []byte, contentType string) (string, error) {
	if f.writeErr != nil {
		return "", f.writeErr
	}
	if f.written == nil {
		f.written = map[string][]byte{}
		f.ctypes = map[string]string{}
	}
	f.written[key] = data
	f.ctypes[key] = contentType
	return "s3://render-bucket/" + key, nil
}

func (f *fakeStore) Read(_ context.Context, uri string) ([]byte, string, error) {
	f.reads = append(f.reads, uri)
	return f.source, f.srcMime, f.readErr
}

func rendererCfg() Config {
	return Config{Binary: CrewKindRenderer, ModelLabel: "renderer", Model: "gemini-3-pro-image", PromptVersion: "v1", RenderBucket: "render-bucket", AgentAppName: "app"}
}

func runRenderer(t *testing.T, root agent.Agent, state map[string]any, plugins []*plugin.Plugin) (string, error) {
	t.Helper()
	ctx := context.Background()
	svc := session.InMemoryService()
	const app, user, sid = "qgen_renderer", "11111111-1111-7111-8111-111111111111:g1", "dispatch:e-1"
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: sid, State: state}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: app, Agent: root, SessionService: svc, PluginConfig: runner.PluginConfig{Plugins: plugins}})
	if err != nil {
		t.Fatal(err)
	}
	var events []*session.Event
	for ev, err := range r.Run(ctx, user, sid, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "BEGIN"}}}, agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		events = append(events, ev)
	}
	return agentdispatch.TerminalText(events, agentdispatch.TerminalAuthor(DispatchRoleRender))
}

func renderState(extra map[string]any) map[string]any {
	st := map[string]any{"tenant_id": "11111111-1111-7111-8111-111111111111", "user_gcid": "g1", "dispatch_idempotency_key": "agent_dispatch.qgen_render.e-1",
		agentdispatch.StateKeyDispatchExecutionID: "e-1", "render_prompt": "A fox explaining fractions with a pizza", "job_id": "job-7", "chunk_id": "chunk-2"}
	for k, v := range extra {
		st[k] = v
	}
	return st
}

func TestRenderer_happyPathWritesTheImageAndAnswersTheEnvelope(t *testing.T) {
	inv := &fakeInvoker{result: modelgatewayclient.ImageResult{Bytes: []byte("PNGBYTES"), MimeType: "image/png", ModelVersion: "gemini-3-pro-image-001", Attempts: 1}}
	store := &fakeStore{}
	root, err := NewRendererAgent(rendererCfg(), inv, store)
	if err != nil {
		t.Fatal(err)
	}
	plugins, err := NewRendererPlugins()
	if err != nil {
		t.Fatal(err)
	}
	out, err := runRenderer(t, root, renderState(nil), plugins)
	if err != nil {
		t.Fatal(err)
	}
	var env RenderEnvelope
	if err := json.Unmarshal([]byte(out), &env); err != nil {
		t.Fatalf("envelope not JSON: %v: %s", err, out)
	}
	if !strings.HasPrefix(env.ImageURI, "s3://render-bucket/tenants/11111111-1111-7111-8111-111111111111/jobs/job-7/") || !strings.HasSuffix(env.ImageURI, ".png") {
		t.Fatalf("image_uri = %q", env.ImageURI)
	}
	if env.MimeType != "image/png" || env.ModelID != "gemini-3-pro-image" || env.ModelVersion != "gemini-3-pro-image-001" || env.PromptVersion != "v1" || env.JobID != "job-7" || env.ChunkID != "chunk-2" || env.Bytes != 8 || env.Edit {
		t.Fatalf("envelope = %+v", env)
	}
	if len(inv.reqs) != 1 || inv.reqs[0].TenantID != "11111111-1111-7111-8111-111111111111" || inv.reqs[0].GCID != "g1" || inv.reqs[0].Prompt != "A fox explaining fractions with a pizza" || inv.reqs[0].DispatchIdempotencyKey != "agent_dispatch.qgen_render.e-1" || inv.reqs[0].ActionCode != "" || len(inv.reqs[0].SourceImage) != 0 {
		t.Fatalf("image request = %+v", inv.reqs[0])
	}
	key := strings.TrimPrefix(env.ImageURI, "s3://render-bucket/")
	if string(store.written[key]) != "PNGBYTES" || store.ctypes[key] != "image/png" {
		t.Fatalf("bucket write wrong: %v", store.written)
	}
}

// The fixture names the CONFIGURED render bucket and the dispatch's OWN tenant
// prefix: it previously pointed at another bucket under tenant "t", which
// authorizeSourceImageURI now refuses. This is the legitimate-edit regression
// guard, so keep it inside the tenant prefix.
func TestRenderer_editReadsTheSourceByReference(t *testing.T) {
	inv := &fakeInvoker{result: modelgatewayclient.ImageResult{Bytes: []byte("JPG"), MimeType: "image/jpeg"}}
	store := &fakeStore{source: []byte("ORIGINAL"), srcMime: "image/png"}
	root, _ := NewRendererAgent(rendererCfg(), inv, store)
	out, err := runRenderer(t, root, renderState(map[string]any{"source_image_uri": "s3://render-bucket/tenants/11111111-1111-7111-8111-111111111111/jobs/j/old.png", "job_id": ""}), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(store.reads) != 1 || string(inv.reqs[0].SourceImage) != "ORIGINAL" || inv.reqs[0].SourceMime != "image/png" {
		t.Fatalf("edit must read the source by s3:// reference: reads=%v req=%+v", store.reads, inv.reqs[0])
	}
	var env RenderEnvelope
	_ = json.Unmarshal([]byte(out), &env)
	if !env.Edit || !strings.Contains(env.ImageURI, "/jobs/e-1/") || !strings.HasSuffix(env.ImageURI, ".jpg") {
		t.Fatalf("edit envelope = %+v (the dispatch execution id scopes the key when job_id is absent)", env)
	}
	// A source that cannot be read is transient (the store may be momentarily unavailable), not permanent.
	// The URI must be AUTHORIZED (own bucket + own tenant prefix) or the run is refused
	// permanently before the read, and this case would stop testing what it names.
	store = &fakeStore{readErr: errors.New("503 backend")}
	root, _ = NewRendererAgent(rendererCfg(), &fakeInvoker{}, store)
	_, err = runRenderer(t, root, renderState(map[string]any{"source_image_uri": "s3://render-bucket/tenants/11111111-1111-7111-8111-111111111111/jobs/j/k.png"}), nil)
	var perm *agentdispatch.PermanentError
	if err == nil || errors.As(err, &perm) {
		t.Fatalf("read failure must be transient: %v", err)
	}
}

func TestRenderer_permanentFaultsBeforeAndAtTheModel(t *testing.T) {
	var perm *agentdispatch.PermanentError
	cases := map[string]struct {
		state map[string]any
		inv   *fakeInvoker
		token string
	}{
		"missing prompt": {state: renderState(map[string]any{"render_prompt": "  "}), inv: &fakeInvoker{}, token: "missing_render_prompt"},
		"unknown mode":   {state: renderState(map[string]any{"mode": "mermaid"}), inv: &fakeInvoker{}, token: "unknown_render_mode"},
		"bad source uri": {state: renderState(map[string]any{"source_image_uri": "https://x/y.png"}), inv: &fakeInvoker{}, token: "invalid_source_image_uri"},
		"blocked image":  {state: renderState(nil), inv: &fakeInvoker{err: modelgatewayclient.ErrNoImage}, token: "image_blocked"},
	}
	for name, tc := range cases {
		root, _ := NewRendererAgent(rendererCfg(), tc.inv, &fakeStore{})
		_, err := runRenderer(t, root, tc.state, nil)
		if !errors.As(err, &perm) || !strings.Contains(err.Error(), tc.token) {
			t.Fatalf("%s: err=%v", name, err)
		}
		if tc.token != "image_blocked" && len(tc.inv.reqs) != 0 {
			t.Fatalf("%s: the model must not be called", name)
		}
	}
	// A throttled gateway (after the invoker's own retries) is a transient run error, retried by the handler.
	root, _ := NewRendererAgent(rendererCfg(), &fakeInvoker{err: errors.New("modelgatewayclient: image Invoke: rpc error: code = Unavailable")}, &fakeStore{})
	if _, err := runRenderer(t, root, renderState(nil), nil); err == nil || errors.As(err, &perm) {
		t.Fatalf("throttling must stay transient: %v", err)
	}
	// A bucket write failure is transient too.
	root, _ = NewRendererAgent(rendererCfg(), &fakeInvoker{result: modelgatewayclient.ImageResult{Bytes: []byte{1}, MimeType: "image/png"}}, &fakeStore{writeErr: errors.New("s3 500")})
	if _, err := runRenderer(t, root, renderState(nil), nil); err == nil || errors.As(err, &perm) {
		t.Fatalf("write failure must stay transient: %v", err)
	}
	if _, err := NewRendererAgent(rendererCfg(), nil, nil); err == nil {
		t.Fatal("nil collaborators must be refused")
	}
}

func TestRenderObjectKeyAndURI(t *testing.T) {
	k := RenderObjectKey("t1", "", "e-9", "image/webp")
	if !strings.HasPrefix(k, "tenants/t1/jobs/e-9/") || !strings.HasSuffix(k, ".webp") {
		t.Fatalf("key = %q", k)
	}
	if k2 := RenderObjectKey("t1", "job", "e-9", "application/octet-stream"); !strings.HasPrefix(k2, "tenants/t1/jobs/job/") || !strings.HasSuffix(k2, ".png") {
		t.Fatalf("unknown mime must default to png: %q", k2)
	}
	if RenderObjectKey("t1", "", "", "") == RenderObjectKey("t1", "", "", "") {
		t.Fatal("keys must be unique")
	}
	b, key, err := splitObjectURI("s3://bucket-a/tenants/t/jobs/j/x.png")
	if err != nil || b != "bucket-a" || key != "tenants/t/jobs/j/x.png" {
		t.Fatalf("split = %q %q %v", b, key, err)
	}
	for _, bad := range []string{"https://bucket/x", "s3://bucket-only", "s3:///x", ""} {
		if _, _, err := splitObjectURI(bad); err == nil {
			t.Fatalf("%q must be refused", bad)
		}
	}
}

func TestLoadRendererConfig_guardsAndDefaults(t *testing.T) {
	t.Setenv("CHORA_PROJECT_ID", "p1")
	t.Setenv("CHORA_GATEWAY_TENANT_ID", "t1")
	t.Setenv("CHORA_GATEWAY_GCID", "g1")
	t.Setenv("QGEN_RENDER_BUCKET", "")
	t.Setenv("QGEN_RENDERER_MODEL", "")
	t.Setenv("QGEN_RENDER_TIMEOUT_SECONDS", "")
	t.Setenv("QGEN_RENDER_RETRY_ATTEMPTS", "")
	t.Setenv("QGEN_RENDER_RETRY_MAX_DELAY_SECONDS", "")
	if _, err := LoadRendererConfig(); err == nil || !strings.Contains(err.Error(), "QGEN_RENDER_BUCKET") {
		t.Fatalf("bucket guard: %v", err)
	}
	t.Setenv("QGEN_RENDER_BUCKET", "chora-ai-assist-images-dev")
	cfg, err := LoadRendererConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Binary != CrewKindRenderer || cfg.Model != "gemini-3-pro-image" || len(cfg.FallbackModels) != 1 || cfg.FallbackModels[0] != "gemini-2.5-flash-image" || cfg.PromptVersion != "v1" || cfg.RenderBucket != "chora-ai-assist-images-dev" || cfg.ImageTimeout != 120*time.Second || cfg.ImageRetry.Attempts != 4 || cfg.ImageRetry.MaxDelay != 30*time.Second || cfg.GatewayCrewKind != CrewSurface {
		t.Fatalf("config = %+v", cfg)
	}
	gw := cfg.GatewayConfig()
	if gw.Surface != "qgen" || gw.AgentID != "qgen_renderer" || gw.LogicalModelID != "gemini-3-pro-image" {
		t.Fatalf("gateway config = %+v", gw)
	}
	attrs := cfg.LogAttrs()
	joined := strings.Builder{}
	for _, a := range attrs {
		joined.WriteString(strings.TrimSpace(strings.ReplaceAll(strings.Trim(jsonString(a), `"`), "\n", " ")) + " ")
	}
	if !strings.Contains(joined.String(), "render_bucket chora-ai-assist-images-dev") || !strings.Contains(joined.String(), "surface qgen") || strings.Contains(joined.String(), "tenancy") {
		t.Fatalf("log attrs: %s", joined.String())
	}
	t.Setenv("QGEN_RENDER_TIMEOUT_SECONDS", "abc")
	if _, err := LoadRendererConfig(); err == nil {
		t.Fatal("a non-numeric timeout must be refused")
	}
	t.Setenv("QGEN_RENDER_TIMEOUT_SECONDS", "45")
	t.Setenv("QGEN_RENDER_RETRY_ATTEMPTS", "2")
	t.Setenv("QGEN_RENDERER_MODEL", "gemini-2.5-flash-image")
	cfg, _ = LoadRendererConfig()
	if cfg.ImageTimeout != 45*time.Second || cfg.ImageRetry.Attempts != 2 || cfg.Model != "gemini-2.5-flash-image" || cfg.FallbackModels[0] != "gemini-2.5-flash-image" {
		t.Fatalf("overrides: %+v", cfg)
	}
}

func jsonString(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestDispatchIdentities(t *testing.T) {
	for binary, want := range map[string][2]string{
		CrewKindQuestion: {DispatchRoleGenerate, "chora-qgen-question"},
		CrewKindCritic:   {DispatchRoleCritique, "chora-qgen-critic"},
		CrewKindRenderer: {DispatchRoleRender, "chora-qgen-renderer"},
	} {
		if DispatchRole(binary) != want[0] || ServiceName(binary) != want[1] {
			t.Fatalf("%s: role %q service %q", binary, DispatchRole(binary), ServiceName(binary))
		}
	}
	if DispatchRole("nope") != "" || ServiceName("nope") != "" {
		t.Fatal("unknown binaries resolve empty")
	}
	if _, err := dispatchServeConfig(Config{Binary: "nope"}, nil, nil); err == nil {
		t.Fatal("an unknown binary must be refused")
	}
	sc, err := dispatchServeConfig(Config{Binary: CrewKindRenderer, AgentAppName: "app"}, nil, nil)
	if err != nil || sc.AgentRole != "qgen_render" || sc.ServiceName != "chora-qgen-renderer" || sc.AppName != "app" || sc.SessionKey != nil {
		t.Fatalf("serve config = %+v %v", sc, err)
	}
	if err := refuseArgs(CrewKindRenderer, []string{"web"}); err == nil || !strings.Contains(err.Error(), "subscriber-only") {
		t.Fatalf("args must be refused: %v", err)
	}
	tc := RendererTerminationConfig()
	if tc.AgentID != "qgen_renderer" || tc.CrewKind != "qgen" || tc.CrewPattern != "P1_SINGLE_AGENT" {
		t.Fatalf("termination = %+v", tc)
	}
	ps, err := NewRendererPlugins()
	if err != nil || len(ps) != 2 || !strings.Contains(ps[0].Name(), "inbound_trace") || !strings.Contains(ps[1].Name(), "termination") {
		t.Fatalf("renderer chain: %d %v", len(ps), err)
	}
}

func TestRunRenderer_composesTheLaneAndSurfacesTheSubscriberError(t *testing.T) {
	t.Setenv("CHORA_PROJECT_ID", "p1")
	t.Setenv("CHORA_GATEWAY_TENANT_ID", "t1")
	t.Setenv("CHORA_GATEWAY_GCID", "g1")
	t.Setenv("QGEN_RENDER_BUCKET", "render-bucket")
	oldT, oldI, oldS, oldServe := initTracing, newImageInvoker, newObjectStore, serveSubscriber
	t.Cleanup(func() { initTracing, newImageInvoker, newObjectStore, serveSubscriber = oldT, oldI, oldS, oldServe })
	initTracing = func(context.Context, string) (func(context.Context) error, error) {
		return func(context.Context) error { return nil }, nil
	}
	newImageInvoker = func(context.Context, Config) (ImageInvoker, error) { return &fakeInvoker{}, nil }
	newObjectStore = func(context.Context, string) (ObjectStore, error) { return &fakeStore{}, nil }
	var got agentdispatch.ServeConfig
	errStop := errors.New("stop")
	serveSubscriber = func(_ context.Context, cfg agentdispatch.ServeConfig, _ agentdispatch.RunOptions) error {
		got = cfg
		return errStop
	}
	if err := RunRenderer(context.Background(), []string{"web", "-port", "8080"}); err == nil || !strings.Contains(err.Error(), "subscriber-only") {
		t.Fatalf("args must be refused before config: %v", err)
	}
	if err := RunRenderer(context.Background(), nil); !errors.Is(err, errStop) {
		t.Fatalf("want the subscriber error to surface, got %v", err)
	}
	if got.AgentRole != "qgen_render" || got.ServiceName != "chora-qgen-renderer" || got.RootAgent == nil || len(got.Plugins.Plugins) != 2 {
		t.Fatalf("serve config = %+v", got)
	}
	newObjectStore = func(context.Context, string) (ObjectStore, error) { return nil, errors.New("no bucket") }
	if err := RunRenderer(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "no bucket") {
		t.Fatalf("store failure must be fatal before serving: %v", err)
	}
	newImageInvoker = func(context.Context, Config) (ImageInvoker, error) { return nil, errors.New("no gateway") }
	if err := RunRenderer(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "no gateway") {
		t.Fatalf("invoker failure must be fatal before serving: %v", err)
	}
}

// A source image is an AUTHORIZATION decision, not just a shape check: the
// dispatch may only edit an object inside the configured render bucket and
// under its OWN tenant prefix. The tenant comes from the dispatch ENVELOPE
// (agentdispatch.Request.SessionState overwrites any input_payload copy), so a
// caller cannot widen its own scope by putting a different tenant_id in the
// body. Every refusal is permanent: no redelivery makes another tenant's
// object readable.
func TestRenderer_refusesASourceOutsideTheBucketOrTenantPrefix(t *testing.T) {
	var perm *agentdispatch.PermanentError
	const self = "11111111-1111-7111-8111-111111111111" // renderState's tenant
	const other = "22222222-2222-7222-8222-222222222222"
	cases := map[string]string{
		"another bucket":           "s3://exports-bucket/tenants/" + self + "/jobs/j/x.png",
		"another tenant, same bkt": "s3://render-bucket/tenants/" + other + "/jobs/j/x.png",
		"no tenant prefix at all":  "s3://render-bucket/exports/nightly.png",
		"tenant prefix confusion":  "s3://render-bucket/tenants/" + self + "EVIL/jobs/j/x.png",
		"traversal out of the pfx": "s3://render-bucket/tenants/" + self + "/../" + other + "/x.png",
		"bucket name as a prefix":  "s3://render-bucket-evil/tenants/" + self + "/jobs/j/x.png",
	}
	for name, uri := range cases {
		inv := &fakeInvoker{}
		store := &fakeStore{source: []byte("ORIGINAL"), srcMime: "image/png"}
		root, _ := NewRendererAgent(rendererCfg(), inv, store)
		_, err := runRenderer(t, root, renderState(map[string]any{"source_image_uri": uri}), nil)
		if !errors.As(err, &perm) || !strings.Contains(err.Error(), "forbidden_source_image_uri") {
			t.Fatalf("%s (%s): want a permanent forbidden_source_image_uri, got %v", name, uri, err)
		}
		if len(store.reads) != 0 {
			t.Fatalf("%s: the object must NOT be read before the check: reads=%v", name, store.reads)
		}
		if len(inv.reqs) != 0 {
			t.Fatalf("%s: the model must not be called", name)
		}
	}
}
