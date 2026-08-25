# The memory guide: usage policy exposed to MCP clients as a readable
# resource, plus the reflect prompt for end-of-session reviews.

GUIDE = """\
# agent-mem usage guide

You are the interpreter of this memory. The structure gives you pointers
and rankings; you supply judgment. Follow these rules.

## Reading

1. `search(query)` finds entry points.
2. `land(ref)` reads a node plus its signpost of destinations.
3. At each node ask: does this content answer my question?
   - Yes: stop and use it.
   - No, but a destination matches my intent: `steer(ref, dest)`.
   - No destination matches: dead end. Report not found or use
     `query_scoped`.

Signposts show the top seven destinations ranked by access frequency and
recency. More exist? `land(ref, page=1)`. Destinations carry an epistemic
status when it matters (contradicted, contested). Weigh that status in
your answer.

## Writing

Triples are sentences. Write them well.

1. Search before writing. If the proposition already exists, do not
   duplicate it. Dedup works on whole propositions, not labels: Docker
   may appear in many true sentences.
2. Verbs are lowercase snake phrases that read as a sentence:
   Postgres persists_with WAL is good. "Uses" or "is-related-to" is not.
   Reuse verbs that already exist in the neighborhood when they fit.
3. Placement: file each fact as deep as it is true, no deeper. Run
   `placement(source, target)` first. Generalizations go on intermediate
   nodes, specifics go to leaves. For `add(under_graph=...)`: leave it
   out for everyday facts, they live at root. Only pass a graph name
   when the fact belongs to a specific domain.
4. Every claim needs provenance. After creating a claim node, link it:
   user --user_asserts--> claim, or agent_inferred if you derived it.
   Run `provenance()` now and then to catch unbacked claims.
5. Never fabricate epistemics. Verbs like corroborated_by belong to the
   verify pass only. Claims start unverified; run `verify()` to check
   them against sources.
6. Delete only on explicit user request (`forget`). Never prune on your
   own initiative.

## What deserves memory

Apply the salience test before any write: would a future session be
worse off not knowing this? Would the user have to repeat themselves?
If no, do not write. Memory is not a transcript.

Worth remembering:

- preference: identity and taste. Themes, tools, OS, communication style,
  dislikes. Highest value. Never verified. User wins.
- fact: stable truths about projects, goals, decisions, constraints.
- event: something that happened at a time. Experiences, incidents,
  milestones. If it starts with "when" or "during", it is an event.
- procedure: reusable success patterns. Only if it worked and will recur.
- claim: a statement about the world that could be checked. Lowest trust.
  Store unverified and let verify() annotate it later.

Always tag nodes with category keywords (food, university, tool, person,
project). Tags are how later searches find nodes whose labels do not
contain the query words.

Not worth remembering:

- conversation mechanics ("the user asked about X").
- ephemeral task state and one-off details with no recurrence.
- anything re-derivable from the environment: code on disk, config files.
  Point at it instead of copying it.
- secrets, credentials, tokens. Ever.

Corrections are valuable: when the user says "actually I use NixOS now",
that is a new fact plus a supersedes link over the stale one.

## When to write

Prefer one reflection at session end over mid-conversation writes.
Replay the session against the salience test once (`memory://reflect`
walks you through it) and write the few triples that matter. Write
mid-conversation only when the user explicitly says "remember this";
those get provenance user_asserts immediately.

## Rank is order, never importance

Access scores decide visit order so hot memories surface first. A low
score never means a memory does not matter or should not be read. All
memories are equal; rank only costs latency.
"""

REFLECT_PROMPT = """\
Review this session and distill what belongs in long-term memory.

Apply the salience test to each candidate:
would a future session be worse off not knowing this? Would the user have
to repeat themselves?

Steps:
1. List candidate memories. Aim for zero to five. Most sessions produce
   none or one.
2. For each, state its type (preference, fact, procedure, event, claim),
   the noun verb noun triple, and why it passes the test.
3. Run search for each triple proposition. Skip anything already known.
4. For survivors: run placement(source, target), then link with
   provenance user_asserts if the user stated it, agent_inferred if you
   derived it. Claims stay unverified for the verify pass.
5. If the user corrected something previously stored, link the new fact
   to the stale node with a supersedes edge.
6. Report what you wrote and what you skipped, with reasons.

Memory is not a transcript. Skip conversation mechanics, ephemeral state,
re-derivable facts, and secrets.
"""


def register_guide_resource(mcp) -> None:
    # Attaches memory://guide and the reflect prompt to an MCP server.

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
