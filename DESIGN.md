# agent_mem — Design Notes

Graph-based memory for AI agents. Storage format inspired by kliae (nodes, named
edges, nested graphs) but a standalone, single-workspace system — not built on kliae.

## Design Principles

1. **The memory layer stores structure and pointers only.** All interpretation —
   relevance judgment, summarization, ranking-by-meaning — happens in the
   agent's LLM at read time. The structure's job is to make the right pointer
   obvious, not to pre-digest the content. This is the deliberate inversion of
   GraphRAG-style systems (which pre-compute understanding offline via LLM
   summaries and embeddings so retrieval is dumb); agent_mem keeps the memory
   layer dumb and cheap, and spends zero extra compute — the agent's existing
   intelligence is the summarizer/ranker/reasoner.
2. **No embeddings. Not yet.** Semantic similarity work is deferred until scale
   demands it.
3. **Parent metadata = references to leaves, not digests of them.** The
   divergence map is `{leaf_id → label + breadth}` pointers (~50 bytes per
   leaf), never generated text. The parent lists its children ranked; it does
   not describe them. Labels cost a fraction of a paragraph summary and can
   never go stale in content — only rank order changes.
4. Pure graph math is free and welcome (e.g. LCA over parent pointers needs no
   LLM, no embeddings). Anything requiring interpretation is deferred to the
   consuming agent.


## Core Model

- **Nodes = nouns** (memories: facts, preferences, procedures, events)
- **Edges = verbs** (named relationships, active/passive pairs)
  - A traversal path concatenates into a readable sentence:
    `[Postgres] --depends_on--> [B-tree indexes] --degrades_without--> [vacuum tuning]`
  - Steering is linguistic: the agent picks the predicate matching its intent.
  - Passive forms give inverse traversal for free (`teaches` / `is-taught-by`).
- **Nested graphs**: hierarchical domains; a node can contain a child graph.
- **Paths are propositions**: every hop is a well-formed claim; DFS through a
  subtree reads as a coherent narrative.

## Hierarchy & Leaf Diversity

Invariant: **non-sibling leaves under the same parent are distinct by
construction** — if two leaves were similar enough to group, they'd have been
merged or made siblings at write time. So the leaf tier carries maximum
divergence: the destinations beneath one parent are vastly different from each
other. (Siblings themselves are the grouped/similar case; divergence is across
the groups.)

- Parent nodes expose a **divergence map**: the *distinct destinations* reachable
  beneath them (not a summary of their content).
- Leaf count is a **pressure gauge**: high fan-out means the parent was created
  too coarse → **promotion-split** (cluster similar leaves into a new
  intermediate sibling). Tree grows a level instead of widening.
- Fan-out budget ~7±2.

## Ranking ("edge breadth")

- Combined score from **node access counts + edge traversal counts**, weighted by
  recency decay.
- Rank is an **index hint only** — it orders retrieval, never filters. All
  memories have equal importance; low rank only costs latency, never visibility.
- Recency tiebreaker ensures fresh leaves appear on signposts immediately and
  sink if never touched.

## Read Path ("the waterslide")

Momentum-based traversal: touching a memory hands you its neighborhood so each
hop is pre-laid out; the agent only steers at forks.

### The read loop

Every node at every level is a full, self-sufficient memory — intermediate
nodes carry their own content plus their divergence map. The slide does not
have to run to the leaves; leaves are where it *can* go deepest, not where it
must.

```
at each node:
  1. DIRECTION: read the signpost (divergence map) → which destination matches intent?
  2. CHECK: is this node's own content sufficient?   ← stopping condition
     ├─ yes → stop, use it
     └─ no  → steer toward best destination, repeat
```

Three exits:

- **Found** (content match): this node's content answers the question.
- **Routed** (signpost match): no answer here, but the map shows where it
  lives — descend without reading deeply.
- **Dead end**: no destination matches and content doesn't satisfy → bail out
  (return not-found / fall back to `query_scoped`). This bounds the worst case.

Contract: **steer by direction, stop by satisfaction, exit by dead-end.**
Typical cost: mid-tree targets land in 1 pass; leaf targets in 2; worst case ~4
(signpost paging or scoped query).

Every read returns a **rideable payload**: node content + typed edge list +
parent/sub-graph pointer + ranked neighbors + divergence map — so the
satisfied-or-descend decision costs zero extra passes either way. Momentum
decays naturally when neighbors are visited/irrelevant — that boundary replaces
arbitrary token budgets.

### Write placement corollary

Because intermediate nodes are real memories, they are write targets too:
file each triple **as deep as it's true, no deeper**. Generalizations live on
intermediate nodes; only specifics descend to leaves.

## Signpost Paging

If n leaves ≥ k visible slots:

- Page 1 shows top-k by breadth × recency; spread across divergent destinations.
- Agent can skip later pages when page-1 destinations already cover intent.
- n itself stays bounded via promotion-splitting, so paging is rare in practice.

## Write Policy

- Agent writes autonomously from conversation, as **(noun, verb, noun)** triples
  filed into the right branch.
- Duplicate defense: search-before-write discipline. Dedup operates on
  *propositions* (sentence-paths), not node labels — "Docker" legitimately
  appears in many true sentences.
- Rollups/divergence maps go stale on write → lazy invalidation: mark ancestors
  dirty, regenerate on access. Cold branches keep cheap stale maps; hot branches
  stay fresh because they're touched.

## Forgetting

Delete ≠ forget. Explicit user-requested deletion removes nodes/edges;
nothing decays away silently. Low-rank only means slower access.

## Access Layer

Architecture:

```
MCP server ──┐
             ├── core lib (load, index, traverse) ──▶ *.json (source of truth)
CLI binary ──┘
```

- Core library owns loading, FTS index, breadth scoring, divergence maps,
  steering logic. MCP and CLI are thin wrappers; future UI is just another
  client of the same JSON files.
- In-memory graph loaded per invocation (workspace stays small for a long
  time); SQLite only if scale ever demands it.
- Agent-only for now. Future: kliae-style UI for the user to view/edit.

### Tool surface — two tiers sharing one core

Access tier follows intent and is granted explicitly by the harness (never
inferred from conversation). This turns the waterslide discipline from a
convention into a capability boundary: the signpost structure cannot be cheated
because enumeration tools simply don't exist at read tier.

**READ tier** (consumers — need answers):

- `search(query)` → entry points, ranked; rideable summaries, not raw hits
- `land(node_id | query)` → node content + signpost (top-k destinations with
  verbs) + sentence-so-far
- `steer(node_id, edge_id)` → take one branch; next payload
- `query_scoped(graph_id, q)` → mediated deep query: server dives, returns
  matched sentences + node ids. Deep access without exploratory traversal —
  the agent can ask, not wander.
  - **LCA context (pure graph math, no LLM/embeddings):** when a scoped query
    matches 2+ nodes, compute their **Lowest Common Ancestor** via parent
    pointers and return the connecting paths through it — the minimal coherent
    subgraph relating the hits. Free to compute; gives multi-entity queries a
    compact "here's how these relate" answer instead of disconnected matches.

Deliberately absent: neighbors(), subtree(), expand(), raw file dumps. A reader
sees signposts, never subtrees.

**WRITE tier** (mutators — need ground truth):

- Everything in READ, plus:
- `neighbors(id)` → full local neighborhood (verification before surgery)
- `subtree(graph_id)` → load a nested domain wholesale
- `write(triples[])` → noun-verb-noun insertion w/ built-in
  search-before-write dedup check
- `forget(node_id)` → explicit deletion only
- `reindex()` → rebuild in-memory structure from files

A writer sees raw structure because it is about to mutate it and must see what
it's mutating. Tier escalation is explicit: a read agent that needs to write is
re-instantiated as a write agent, leaving a clean boundary in the transcript
(and a clean audit story).

## Epistemics

Truth and memory are different axes. The system stores what was said; it
annotates what is known. Never gatekeep, never delete for being false.

### Provenance as edges

Every claim decomposes into two sentences via the normal grammar:

```
[User] --asserts--> [Earth is flat]     ← always TRUE, first-class, storable
[Earth] --is-shape-of--> [flat]         ← proposition, epistemic_status: unverified
```

Provenance edge types: `user_asserts`, `agent_inferred`, `source_says`.
Every leaf traces back to at least one of these.

Authority typing (per-edge-type trust rules):

- **User domain** (preferences, projects, what they said): user wins, no
  verification ever.
- **World domain** (science, history): user assertions stored but flagged;
  only the system writes corroboration.

Manipulation resistance: a user can lie, but cannot flip
`contradicted_by` → `corroborated_by`. User writes create provenance; only the
system's lookup pass creates corroboration.

### Verification = async annotation, not admission control

Writes land immediately (as unverified). An async background pass picks up
lookupable world-claims and runs a SERP-level check — the search engine's own
verdict (knowledge panels, snippets, ranking), never site content. Cheap,
near-deterministic, non-agentic, hard to game. A contradiction is not a
rejection; it's a typed edge:

```
[claim: flat earth] --contradicted_by--> [web-source: oblate spheroid]
```

Status enum:

```
user_asserted           ← set at write time (provenance)
corroborated_by[srcs]   ← SERP agreed
contradicted_by[srcs]   ← SERP disagreed
contested               ← mixed evidence; needs reasoning to resolve
unverifiable            ← not a world claim / nothing to look up
```

Time-indexed claims ("X is best practice"): annotation carries an as-of date;
truth has a shelf life.

### Complex fallible claims

When SERP evidence is mixed/ambiguous, the pass stamps `contested` and stops.
No agentic deep-check by default — verifying hard claims becomes retrieval-time
work done by the consuming agent (which has full conversational context),
distributed over the times the claim actually matters, instead of write-time
work paid on every claim including ones nobody re-reads.

Plural viewpoints are stored honestly as edges:
`--some_sources_say--> X`, `--other_sources_say--> Y`.

### Retrieval-time behavior

Signposts show epistemic status inline (`unverified · contradicted`). The agent
steers accordingly; if asked directly it reports calibrated recall: "you told me
this; sources disagree." Not amnesia, not obedience — annotation plus framing.

## Open Questions

- Implementation language for core lib (Python for iteration speed vs Rust to
  eventually embed in a Tauri UI).
- Rank formula details (breadth × recency decay constants).
- Promotion-split trigger mechanics and clustering method.

## Status

Brainstorming. Nothing built yet.
