package boot

// Dispatch identity for the ADR-253/254 dispatch lanes. Three strings per
// binary and they are deliberately different, exactly as on every other lane:
//
//	CrewKind      qgen_question / qgen_critic / qgen_renderer   the ADK agent name / termination id
//	DispatchRole  qgen_generate / qgen_critique / qgen_render   what the kennel keys topics on
//	ServiceName   chora-qgen-question / -critic / -renderer     the Deployment; names the subscription
//
// The request subscriptions (chora-qgen-question.agent-dispatch-qgen-generate-requested,
// chora-qgen-critic.agent-dispatch-qgen-critique-requested,
// chora-qgen-renderer.agent-dispatch-qgen-render-requested) are set explicitly by
// the Deployments; the derived name is the fallback.
const (
	DispatchRoleGenerate = "qgen_generate"
	DispatchRoleCritique = "qgen_critique"
	DispatchRoleRender   = "qgen_render"
)

// DispatchRole is the dispatch role for a binary, or "" when unknown. Empty
// is refused by the boot rather than guessed.
func DispatchRole(binary string) string {
	switch binary {
	case CrewKindQuestion:
		return DispatchRoleGenerate
	case CrewKindCritic:
		return DispatchRoleCritique
	case CrewKindRenderer:
		return DispatchRoleRender
	}
	return ""
}

// ServiceName is the Kubernetes Deployment backing a binary.
func ServiceName(binary string) string {
	switch binary {
	case CrewKindQuestion:
		return "chora-qgen-question"
	case CrewKindCritic:
		return "chora-qgen-critic"
	case CrewKindRenderer:
		return "chora-qgen-renderer"
	}
	return ""
}
