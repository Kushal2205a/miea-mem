# mi∫a

**miea** — graph memory for AI agents. *You ∃ in your agent's memory.*

**Structure and pointers only** — all interpretation happens in the consuming
agent's LLM at read time. No pre-computed summaries; embeddings only as an
optional derived cache. Design notes: [DESIGN.md](./DESIGN.md) (local).

## Model

- **Nodes = nouns**, named content records
- **Edges = verbs** (`persists_with`, `contradicts`, …) — traversal paths
  concatenate into sentences: `[Postgres] --persists_with--> [WAL]`
- **Nested graphs**: containment hierarchy; every node at any depth is a full
  memory, not a folder
- **Signposts**: parents list their children as ranked *references*
  (label + breadth score), never generated summaries

## Quick start

```bash
uv tool install --with sentence-transformers .   # from this repo
miea setup                                       # wizard: path + name → MCP JSON
```

Or manually:

```bash
miea init MyMemory --root ~/Documents/my_memory
miea --root ~/Documents/my_memory add Postgres --content "relational database"
miea --root ~/Documents/my_memory link Postgres persists_with WAL
miea --root ~/Documents/my_memory land Postgres      # rideable payload + signpost
miea --root ~/Documents/my_memory steer Postgres WAL # one branch of the slide
```

## Architecture

```
JSON files (durable truth) → in-memory graph/indexes → agent via CLI/MCP
```

Files are never grepped or dumped raw — the core returns shaped payloads only.
Ranking (breadth × recency) orders results but never filters: all memories are
equal in importance; low rank costs latency, never visibility.

## Status

Core complete (47 tests): model, store, waterslide read loop with signpost
paging, write tier with guardrails, LCA, promotion-split, provenance audit,
epistemic verify pass, hybrid FTS+vector search, `wakeup` snapshot, setup
wizard, CLI + MCP server with read/write tiers.
