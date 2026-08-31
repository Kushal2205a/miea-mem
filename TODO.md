# miea — TODO

- [ ] Reconcile manifest.graph_ids with graphs/*.json on load (drift hardening)
- [ ] SERP verifier backend wiring for live epistemics
- [ ] kliae-style UI viewer for memory workspaces
- [ ] Auto-commit hook for workspaces (commit on writes)
- [ ] Verify OpenCode applies tagging discipline from updated guide
- [ ] Calibrated refusal rule in the guide: a route or slide match is a pointer, not an answer. If the landed content does not state the asked fact, say you do not know. Absence is a valid answer. Never infer a user attribute from an unrelated memory. (Case: no country memory stored, "what is my country" must not answer from a japanese dish match)
- [ ] Evidence signal in route() result: expose how strong the best match was (kw_best token overlap, vec_best cosine) next to matched/ambiguous, so the agent can clarify instead of confabulate. Weak match on "what is my country" should return "I don't have your country on record, closest thing is japanese food, is that what you meant"
