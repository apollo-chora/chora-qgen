package boot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/apollo-chora/chora-adk-common/agentdispatch"
	"github.com/apollo-chora/chora-adk-common/modelgatewayclient"
	"github.com/apollo-chora/chora-common/objectstore"
)

// qgen_render payload keys (ADR-254 D6 addendum, binding with the binary):
//
//	render_prompt      REQUIRED  the scene description the kennel composed (the image_spec source)
//	mode               optional  "scene" (absent = scene); anything else is a permanent unknown_render_mode
//	source_image_uri   optional  s3:// object of the CURRENT image for an EDIT (ADR-210 image-to-image)
//	source_image_mime  optional  its mime (default from the object, else image/png)
//	job_id             optional  the qgen job id; the object key prefix (else the dispatch execution id)
//	chunk_id           optional  echoed on the completion
//
// Completion output_payload: {image_uri, mime_type, model_id, model_version,
// prompt_version, job_id, chunk_id, bytes, edit}.
const (
	stateKeyRenderPrompt    = "render_prompt"
	stateKeyRenderMode      = "mode"
	stateKeySourceImageURI  = "source_image_uri"
	stateKeySourceImageMime = "source_image_mime"
	stateKeyJobID           = "job_id"
	stateKeyChunkID         = "chunk_id"
	stateKeyTenantID        = "tenant_id"
	stateKeyUserGCID        = "user_gcid"
	stateKeyDispatchKey     = "dispatch_idempotency_key"
	renderModeScene         = "scene"
)

type stateReader interface {
	Get(string) (any, error)
}

// ImageInvoker is the gateway image call (modelgatewayclient.ImageInvoker in
// production; a fake in tests).
type ImageInvoker interface {
	Invoke(ctx context.Context, req modelgatewayclient.ImageRequest) (modelgatewayclient.ImageResult, error)
}

// ObjectStore is the render bucket: Write puts an image and returns its
// object-store URI; Read fetches the source image of an edit by URI.
type ObjectStore interface {
	Write(ctx context.Context, key string, data []byte, contentType string) (uri string, err error)
	Read(ctx context.Context, uri string) (data []byte, contentType string, err error)
}

// RenderRequest is the parsed qgen_render payload.
type RenderRequest struct {
	Prompt          string
	SourceImageURI  string
	SourceImageMime string
	JobID           string
	ChunkID         string
}

// ParseRenderRequest reads and validates the payload; every fault is
// permanent (no redelivery can supply a prompt or fix a mode).
func ParseRenderRequest(st stateReader) (RenderRequest, error) {
	prompt := strings.TrimSpace(StateString(st, stateKeyRenderPrompt))
	if prompt == "" {
		return RenderRequest{}, agentdispatch.Permanent("missing_render_prompt: the qgen_render dispatch carries the scene description under render_prompt", nil)
	}
	if mode := strings.ToLower(strings.TrimSpace(StateString(st, stateKeyRenderMode))); mode != "" && mode != renderModeScene {
		return RenderRequest{}, agentdispatch.Permanent("unknown_render_mode: "+mode+" (scene is the only dispatched render; Kroki stays in the kennel, ADR-254 D12)", nil)
	}
	src := strings.TrimSpace(StateString(st, stateKeySourceImageURI))
	if src != "" && !strings.HasPrefix(src, "s3://") {
		return RenderRequest{}, agentdispatch.Permanent("invalid_source_image_uri: not an s3:// reference", fmt.Errorf("source_image_uri=%q", src))
	}
	return RenderRequest{
		Prompt:          prompt,
		SourceImageURI:  src,
		SourceImageMime: strings.TrimSpace(StateString(st, stateKeySourceImageMime)),
		JobID:           strings.TrimSpace(StateString(st, stateKeyJobID)),
		ChunkID:         strings.TrimSpace(StateString(st, stateKeyChunkID)),
	}, nil
}

// authorizeSourceImageURI is the AUTHORIZATION half of an edit. ParseRenderRequest
// proves the reference is SHAPED like an s3:// URI; this proves the dispatch is
// allowed to read THAT object. Without it the read primitive below reaches any
// object the renderer's credentials can view, which crosses the tenant boundary
// that the write path already respects through RenderObjectKey: writes are
// partitioned by tenant, so reads must be too.
//
// tenantID MUST come from the dispatch ENVELOPE. agentdispatch Request.SessionState
// stamps state["tenant_id"] from the envelope AFTER merging input_payload, so a
// body copy cannot win; reading the tenant from anywhere else would inherit the
// same caller-supplied weakness this check closes.
func authorizeSourceImageURI(src, renderBucket, tenantID string) error {
	tenantID = strings.TrimSpace(tenantID)
	renderBucket = strings.TrimSpace(renderBucket)
	if tenantID == "" || renderBucket == "" {
		return agentdispatch.Permanent("forbidden_source_image_uri: an edit needs both a dispatch tenant and a configured render bucket", nil)
	}
	bucket, key, err := splitObjectURI(src)
	if err != nil {
		return agentdispatch.Permanent("forbidden_source_image_uri: "+err.Error(), nil)
	}
	if bucket != renderBucket {
		return agentdispatch.Permanent("forbidden_source_image_uri: source bucket is not the render bucket",
			fmt.Errorf("bucket=%q want=%q", bucket, renderBucket))
	}
	// Object keys are literal, so ".." is not resolved by the server; it is
	// refused anyway because no legitimate render key contains one and reasoning
	// about literal-vs-resolved traversal is exactly where this class of bug hides.
	if strings.Contains(key, "..") {
		return agentdispatch.Permanent("forbidden_source_image_uri: object key contains a traversal segment",
			fmt.Errorf("key=%q", key))
	}
	// The trailing slash is load-bearing: without it tenant "…111" would match
	// a key under "…111EVIL/".
	if prefix := "tenants/" + tenantID + "/"; !strings.HasPrefix(key, prefix) {
		return agentdispatch.Permanent("forbidden_source_image_uri: object is outside the dispatch tenant's prefix",
			fmt.Errorf("key=%q want prefix %q", key, prefix))
	}
	return nil
}

var mimeExt = map[string]string{"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif"}

// extMime is the reverse of mimeExt: the object key's extension back to the
// image mime. The object store does not return a stored Content-Type on read,
// so the key extension (written from the same map) is the source of truth.
var extMime = func() map[string]string {
	m := make(map[string]string, len(mimeExt))
	for mime, ext := range mimeExt {
		m[ext] = mime
	}
	return m
}()

// mimeFromExt infers the content type from the object key's extension,
// defaulting to application/octet-stream for an unknown one.
func mimeFromExt(key string) string {
	ext := key[strings.LastIndex(key, ".")+1:]
	if mime, ok := extMime[strings.ToLower(ext)]; ok {
		return mime
	}
	return "application/octet-stream"
}

// RenderObjectKey is the bucket key: tenants/{tenant}/jobs/{job or dispatch execution}/{uuid}.{ext},
// the same prefix the kennel's uploader used, so its signing and retention
// rules keep applying.
func RenderObjectKey(tenantID, jobID, executionID, mime string) string {
	scope := strings.TrimSpace(jobID)
	if scope == "" {
		scope = strings.TrimSpace(executionID)
	}
	if scope == "" {
		scope = "unscoped"
	}
	ext := mimeExt[strings.ToLower(strings.TrimSpace(mime))]
	if ext == "" {
		ext = "png"
	}
	return fmt.Sprintf("tenants/%s/jobs/%s/%s.%s", strings.TrimSpace(tenantID), scope, strings.ReplaceAll(uuid.Must(uuid.NewV7()).String(), "-", ""), ext)
}

// RenderEnvelope is the completion output_payload.
type RenderEnvelope struct {
	ImageURI      string `json:"image_uri"`
	MimeType      string `json:"mime_type"`
	ModelID       string `json:"model_id"`
	ModelVersion  string `json:"model_version,omitempty"`
	PromptVersion string `json:"prompt_version"`
	JobID         string `json:"job_id,omitempty"`
	ChunkID       string `json:"chunk_id,omitempty"`
	Bytes         int    `json:"bytes"`
	Edit          bool   `json:"edit"`
}

// render performs one dispatch: parse, optional source read, gateway image
// Invoke, bucket write, envelope.
func render(ctx context.Context, cfg Config, invoker ImageInvoker, store ObjectStore, st stateReader, executionID string) (RenderEnvelope, error) {
	req, err := ParseRenderRequest(st)
	if err != nil {
		return RenderEnvelope{}, err
	}
	tenantID := strings.TrimSpace(StateString(st, stateKeyTenantID))
	gcid := strings.TrimSpace(StateString(st, stateKeyUserGCID))
	imageReq := modelgatewayclient.ImageRequest{
		TenantID: tenantID, GCID: gcid, Prompt: req.Prompt,
		DispatchIdempotencyKey: strings.TrimSpace(StateString(st, stateKeyDispatchKey)),
		// ADR-254 D7: the renderer inherits the image call's off-meter status.
		ActionCode: "",
	}
	if req.SourceImageURI != "" {
		if err := authorizeSourceImageURI(req.SourceImageURI, cfg.RenderBucket, tenantID); err != nil {
			return RenderEnvelope{}, err
		}
		data, mime, err := store.Read(ctx, req.SourceImageURI)
		if err != nil {
			return RenderEnvelope{}, fmt.Errorf("qgen_renderer: read source image %s: %w", req.SourceImageURI, err)
		}
		if len(data) == 0 {
			return RenderEnvelope{}, agentdispatch.Permanent("empty_source_image: "+req.SourceImageURI+" holds no bytes", nil)
		}
		imageReq.SourceImage = data
		imageReq.SourceMime = req.SourceImageMime
		if imageReq.SourceMime == "" {
			imageReq.SourceMime = mime
		}
	}
	res, err := invoker.Invoke(ctx, imageReq)
	if err != nil {
		if errors.Is(err, modelgatewayclient.ErrNoImage) {
			// A content decision (safety block, text-only answer) does not
			// change on redelivery: report it by name, once.
			return RenderEnvelope{}, agentdispatch.Permanent("image_blocked: "+err.Error(), err)
		}
		return RenderEnvelope{}, fmt.Errorf("qgen_renderer: image invoke: %w", err)
	}
	key := RenderObjectKey(tenantID, req.JobID, executionID, res.MimeType)
	uri, err := store.Write(ctx, key, res.Bytes, res.MimeType)
	if err != nil {
		return RenderEnvelope{}, fmt.Errorf("qgen_renderer: write %s: %w", key, err)
	}
	return RenderEnvelope{
		ImageURI: uri, MimeType: res.MimeType, ModelID: cfg.Model, ModelVersion: res.ModelVersion,
		PromptVersion: cfg.PromptVersion, JobID: req.JobID, ChunkID: req.ChunkID, Bytes: len(res.Bytes),
		Edit: req.SourceImageURI != "",
	}, nil
}

// NewRendererAgent builds the qgen_renderer root: a custom agent (no LLM
// turn) whose single event carries the envelope as its text, which
// agentdispatch publishes as the completion (last text event wins).
func NewRendererAgent(cfg Config, invoker ImageInvoker, store ObjectStore) (agent.Agent, error) {
	if invoker == nil || store == nil {
		return nil, errors.New("NewRendererAgent: image invoker and object store are required")
	}
	return agent.New(agent.Config{
		Name:        CrewKindRenderer,
		Description: "qgen_render: the generative scene image per chunk, written to the render bucket",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				st := ctx.Session().State()
				env, err := render(ctx, cfg, invoker, store, st, StateString(st, agentdispatch.StateKeyDispatchExecutionID))
				if err != nil {
					yield(nil, err)
					return
				}
				b, _ := json.Marshal(env)
				ev := session.NewEvent(ctx.InvocationID())
				ev.Author = CrewKindRenderer
				ev.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{Text: string(b)}}}
				yield(ev, nil)
			}
		},
	})
}

// S3ObjectStore is the production ObjectStore over one bucket: an S3-compatible
// endpoint (MinIO locally) via chora-common/objectstore, configured from the
// environment (S3_ENDPOINT / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_REGION).
type S3ObjectStore struct {
	store  *objectstore.Store
	bucket string
}

// NewS3ObjectStore opens the bucket client from the environment (static
// credentials; the endpoint holds objectCreator + objectViewer on the render
// bucket).
func NewS3ObjectStore(ctx context.Context, bucket string) (*S3ObjectStore, error) {
	if strings.TrimSpace(bucket) == "" {
		return nil, errors.New("NewS3ObjectStore: bucket required")
	}
	store, err := objectstore.New(objectstore.ConfigFromEnv())
	if err != nil {
		return nil, fmt.Errorf("objectstore.New: %w", err)
	}
	return &S3ObjectStore{store: store, bucket: strings.TrimSpace(bucket)}, nil
}

// Write uploads the image and returns its s3:// URI.
func (s *S3ObjectStore) Write(ctx context.Context, key string, data []byte, contentType string) (string, error) {
	if err := s.store.Put(ctx, key, bytes.NewReader(data), int64(len(data)), contentType); err != nil {
		return "", err
	}
	return "s3://" + s.bucket + "/" + key, nil
}

// Read fetches an object by s3:// URI (any bucket the credentials can view).
func (s *S3ObjectStore) Read(ctx context.Context, uri string) ([]byte, string, error) {
	bucket, key, err := splitObjectURI(uri)
	if err != nil {
		return nil, "", err
	}
	if bucket != s.bucket {
		return nil, "", fmt.Errorf("s3 uri bucket %q != configured %q", bucket, s.bucket)
	}
	rc, err := s.store.Get(ctx, key)
	if err != nil {
		return nil, "", err
	}
	defer rc.Close()
	data, err := io.ReadAll(rc)
	if err != nil {
		return nil, "", err
	}
	return data, mimeFromExt(key), nil
}

// splitObjectURI splits s3://bucket/object into its parts.
func splitObjectURI(uri string) (bucket, key string, err error) {
	rest, ok := strings.CutPrefix(strings.TrimSpace(uri), "s3://")
	if !ok {
		return "", "", fmt.Errorf("not an s3:// URI: %q", uri)
	}
	bucket, key, ok = strings.Cut(rest, "/")
	if !ok || bucket == "" || key == "" {
		return "", "", fmt.Errorf("s3:// URI missing bucket or object: %q", uri)
	}
	return bucket, key, nil
}
