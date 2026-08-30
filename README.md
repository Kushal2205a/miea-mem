<div align="center">

# mi∃a

</div>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/miea-mem)](https://pypi.org/project/miea-mem/)
[![License](https://img.shields.io/github/license/Kushal2205a/miea-mem)](LICENSE)

</div>

mi∃a is graph memory for AI agents. A memory is a node holding a fact,
connected to other nodes by named edges. Everything is stored as one
readable JSON file per node, edge and graph. The agent queries them over
MCP and steers through the graph at read time. The
approach assumes the consuming LLM does the interpretation work well, so
the storage layer stays dumb and cheap. It holds structure and pointers.


## Install

```bash
uv tool install miea-mem
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Setup

```bash
miea setup
```

The wizard asks for a workspace folder and your name, creates the
workspace, initializes git history in it, and prints an MCP config block.
Paste that block into any MCP client (OpenCode, Claude Code, Cursor) and
restart the agent.

## Usage

Tell your agent things and it stores them.

> remember that I like biryani

Later, ask it questions.

> what food do I like?

The agent runs `search` to find an entry node, reads its content and signpost with `land`, then picks a direction with `route` — a hybrid match over the fork's branch entries — and rides the chosen branch in one pass with `slide`, checking sufficiency against the proposition chain on arrival.

A landed node looks like this.

```
* [Kushal] (anchor)
  first: 2026-08-23
  epistemic: unverifiable
  destinations:
    [Biryani] <--likes-- (score 2.00)
    [Blade Runner 2049] <--likes-- (score 2.00)
    [Woxsen University 2022-2026] <--studied_at-- (score 2.69)
```

You can also drive it from the terminal.

```bash
miea --root ~/Documents/my-memory add Postgres --content "relational database"
miea --root ~/Documents/my-memory link Postgres persists_with WAL
miea --root ~/Documents/my-memory land Postgres
miea --root ~/Documents/my_memory search relational
```

## How it works

Storage is one JSON file per node, edge and graph. On startup these load
into flat dicts plus three derived indexes which are word counts for keyword search,
adjacency lists for traversal, and a containment map for nesting.

Reading a node returns its content and a signpost of destinations. Each
destination carries the verb of the edge that reaches it and an access
score. Scores order the list. Nothing is hidden, paging and scoped queries
reach everything stored.

Fork nodes resolve their branch tier through a stored divergence map of
pointers — one entry per child branch: the branch's lowest common
ancestor (an anchor, or the leaf itself for a singleton) plus a cue to its
hottest leaf. The map is stored in the node file but ordered live from
breadth at read time, and regenerates lazily when a structural write marks
it stale. With a query in hand, `route` hybrid-matches across the branch
entries; `slide` then rides the chosen branch in one pass, returning the
noun-verb-noun chain and any slid-past nodes so sufficiency is judged on
arrival.

Writing takes noun verb noun triples. Duplicate triples do nothing. Verbs
must be lowercase snake phrases. A small set of epistemic verbs is
reserved so stored claims can only be corroborated or contradicted by the
verify pass, which checks pending claims against search results when you
run it.

## Design rules

- The consuming agent does all interpretation. This package stores
  structure and pointers.
- Workspace files are plain JSON, readable and editable by hand. They are
  the only source of truth. Derived indexes rebuild from them at any time.
- Access counts rank results without ever filtering them. Every memory
  stays reachable, low rank only costs latency.
- Deletion is explicit. Nothing decays or disappears on its own.

## Performance

Timings on a Ryzen 7 laptop with NVMe storage, Linux, p50 over repeated
runs, keyword search only. Workspaces use a realistic topology where
fanout stays near the split cap. Rerun yourself with
`uv run python bench/run.py 1000 10000`; The exact numbers will differ machine to machine, but the ratios between operations stay the same..

| Operation | 1,000 nodes | 10,000 nodes |
|---|---|---|
| cold load | 47 ms | 136 ms |
| search | 4.9 ms | 14.6 ms |
| land (anchor) | 7.7 ms | 22.1 ms |
| steer | 7.9 ms | 24.6 ms |
| route (direction) | 3.3 ms | 10.7 ms |
| slide (descent) | 7.8 ms | 24.0 ms |
| write new triple | 7.8 ms | 22.4 ms |
| write duplicate | 0.3 ms | 0.7 ms |
| resident memory | 30 MB | 30 MB |

Reads and writes stay fast because everything happens in memory and the
files are written through. Direction picking (`route`) scores only the
fork's branch entries plus their cue leaves, so it is cheaper than a
single `land`; the one-pass `slide` costs about the same as one `steer`
while covering the whole descent. Semantic search trades speed for
recall — it finds memories by meaning when the query shares no words with
them, and it costs roughly 0.8 seconds per query at ten thousand nodes
because vectors are compared one by one.

Cold load grows linearly with file count since every entity is its own
JSON file, which is the main limit of this design. The numbers above come
from a 10,000 node workspace. Workspaces that size stay comfortable,
bigger ones slow down at startup first.


## CLI reference

The commands you will touch most.

```bash
miea search "what should I look at first"
miea land Kushal
miea route Postgres "vacuum tuning"
miea slide "Postgres" "Postgres: unlinked" --deep
miea link Postgres persists_with WAL
miea suggest-split Postgres "vacuum tuning"
miea verify
```

The full command list, including wakeup, placement, provenance and
forget, lives in [SETUP.md](SETUP.md). MCP tools mirror these operations
under two tiers. Readers get search, land, steer, scoped queries and LCA.
Writers also get add, link, forget, neighbors, placement, verify and
provenance.

## Development

```bash
git clone https://github.com/Kushal2205a/miea-mem.git
cd miea-mem
uv sync
uv run pytest
```

## License

MIT
