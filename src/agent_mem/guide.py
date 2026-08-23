"""The memory guide: write-policy resource exposed to MCP clients.

Etiquette lives here as a readable resource; enforcement lives in the tools.
"""

GUIDE = """\
# agent_mem — usage guide

You are the interpreter of this memory. The structure gives you pointers and
rankings; you supply judgment. Follow these rules.

## Reading (always)

1. `search(query)` → pick an entry point.
2. `land(ref)` → read content + signpost.
3. At each node ask: does this content satisfy me?
   - YES → stop, use it.
   - NO, but a signpost destination matches my intent → `steer(ref, dest)`.
   - NO destination matches → dead end: report not-found or `query_scoped`.

Signposts show top-7 destinations ranked by breadth×recency. More exist?
`land(ref, page=1)`. Destinations carry epistemic status when noteworthy
(contradicted / contested / unverified) — weigh that in your answer.

## Writing (write tier only)

Triples are sentences. Write them well:

1. **Search before writing.** If the proposition already exists, do not
   duplicate it. Dedup operates on whole propositions, not labels — "Docker"
   may legitimately appear in many true sentences.
2. **Verbs are lowercase_snake phrases** that read as a sentence:
   `[Postgres] --persists_with--> [WAL]` ✓
   `[Postgres] --Uses--> [WAL]`, `--is-related-to-->` ✗
   Reuse existing verbs from the neighborhood when they fit.
3. **Placement: as deep as it's true, no deeper.** Run `placement(source,
   target)`; file generalizations at intermediate nodes, specifics at leaves.
4. **Every claim needs provenance.** After creating a claim-type node, link it:
   `[user] --user_asserts--> [claim]` (or `agent_inferred` if you derived it).
   Run `provenance()` occasionally to catch unbacked claims.
5. **Never fabricate epistemics.** Verbs like `corroborated_by` are
   system-reserved; only the verify pass writes them. Claims start
   unverified; run `verify()` to annotate them against sources.
6. **Delete only on explicit user request** (`forget`). Never prune
   "to clean up" on your own initiative.

## Rank is order, never importance

Breadth scores decide visit order so hot memories surface first. A low score
never means a memory doesn't matter or shouldn't be read — all memories are
equal; rank only costs latency.
"""


def register_guide_resource(mcp) -> None:
    """Attach memory://guide to an MCPServer instance."""
    from mcp.server.mcpserver import MCPServer  # noqa: F401 — type reference

    @mcp.resource("memory://guide", name="Memory Usage Guide",
                  description="How to read/write this agent's memory. Read "
                              "once before your first write.")
    def guide() -> str:
        return GUIDE
