"""The memory guide: write-policy resource exposed to MCP clients.

Etiquette lives here as a readable resource; enforcement lives in the tools.
"""

GUIDE = """\
# miea — usage guide

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

## What deserves memory

Ask the salience test before any write:
**"Would a future session be worse off not knowing this? Would the user have
to repeat themselves?"**
If no → don't write. Memory is not a transcript.

Worth remembering, by type:

- `preference` — user identity & taste: themes, tools, OS, communication
  style, dislikes. Highest value; never verify; user wins.
- `fact` (user-domain) — projects, goals, decisions, constraints: "building
  kliae", "chose Python for miea".
- `procedure` — reusable success patterns: "deploy: uv sync then restart
  service". Only if it worked and will recur.
- `event` — dated context: "migrated to uv on 2026-08-23".
- `claim` — world facts ("X causes Y"): lowest trust; write unverified, let
  `verify()` annotate them later.

NOT worth remembering:

- Conversation mechanics ("user asked about X"), ephemeral task state,
  one-off details with no recurrence.
- Anything re-derivable from the environment (code on disk, config files) —
  point at it instead of copying it.
- Secrets, credentials, tokens — never.

Corrections are gold: when the user says "actually I use NixOS now", that's a
new fact PLUS a superseding relationship over the stale one — link both.

## When to write

Prefer the **end-of-session reflection**: replay the conversation against the
salience test once, then emit the few triples that matter (`memory://reflect`
walks you through it). Write mid-conversation only when the user explicitly
says "remember this" — those go in immediately with provenance
`user_asserts`.

## Rank is order, never importance

Breadth scores decide visit order so hot memories surface first. A low score
never means a memory doesn't matter or shouldn't be read — all memories are
equal; rank only costs latency.
"""


REFLECT_PROMPT = """\
Review this session and distill what belongs in long-term memory.

Apply the salience test to each candidate:
"Would a future session be worse off not knowing this? Would the user have \
to repeat themselves?"

Steps:
1. List candidate memories (aim for 0-5; most sessions yield none or one).
2. For each, state its type (preference / fact / procedure / event / claim), \
the (noun, verb, noun) triple, and why it passes the test.
3. Run `search` for each triple's proposition — skip anything already known.
4. For survivors: run `placement(source, target)`, then `link(...)` with \
provenance="user_asserts" if the user stated it, "agent_inferred" if you \
derived it. Claims additionally stay unverified for the `verify` pass.
5. If the user corrected something previously stored, link the new fact to \
the stale node with a `supersedes` edge instead of leaving both unconnected.
6. Report what you wrote and what you deliberately skipped (and why).

Remember: memory is not a transcript. Skip conversation mechanics, ephemeral \
state, re-derivable facts, and secrets.
"""


def register_guide_resource(mcp) -> None:
    """Attach memory://guide and memory://reflect to an MCPServer instance."""
    @mcp.resource("memory://guide", name="Memory Usage Guide",
                  description="How to read/write this agent's memory. Read "
                              "once before your first write.")
    def guide() -> str:
        return GUIDE

    @mcp.prompt(name="reflect", title="Session reflection",
                description="Distill this session into memory writes. Call "
                            "at session end.")
    def reflect() -> str:
        return REFLECT_PROMPT
