# agent_mem

Graph-based memory for AI agents. **Structure and pointers only** — all
interpretation happens in the consuming agent's LLM at read time. No embeddings,
no pre-computed summaries. Design doc: [DESIGN.md](./DESIGN.md).

## Model

- **Nodes = nouns**, named content records
- **Edges = verbs** (`persists_with`, `contradicts`, …) — traversal paths
  concatenate into sentences: `[Postgres] --persists_with--> [WAL]`
- **Nested graphs**: containment hierarchy; every node at any depth is a full
  memory, not a folder
- **Divergence maps**: parents list their children as ranked *references*
  (label + breadth score), never generated summaries

## Quick start

```bash
uv sync
uv run agent-mem init MyMemory --root ~/agent_mem_ws
export AGENT_MEM_ROOT=~/agent_mem_ws

agent-mem add Postgres --content "relational database"
agent-mem link Postgres persists_with WAL     # creates missing nodes
agent-mem land Postgres                       # rideable payload + signpost
agent-mem steer Postgres WAL                  # one branch of the slide
agent-mem search relational                   # entry points, ranked
agent-mem query-scoped MyMemory vacuum        # mediated deep dive
agent-mem lca WAL B-trees                     # lowest common ancestor
agent-mem forget WAL                          # explicit deletion only
```

## Architecture

```
JSON files (durable truth) → in-memory graph/indexes (this lib) → agent via CLI/MCP
```

Files are never grepped or dumped raw — the core returns shaped payloads only.
Ranking (breadth × recency) orders results but never filters: all memories are
equal in importance; low rank costs latency, never visibility.

## Status

Early. Core model, read loop, write tier, LCA, and CLI work; MCP server next.
