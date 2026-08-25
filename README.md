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

Under the hood the agent runs `search` to find an entry node, reads its
content and signpost with `land`, then follows named edges with `steer`
until it has the answer.

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
`uv run python bench/run.py 1000 10000`; absolute numbers vary by machine,
the ratios hold.

| Operation | 1,000 nodes | 10,000 nodes |
|---|---|---|
| cold load | 46 ms | 494 ms |
| search | 5.2 ms | 52.6 ms |
| land (anchor) | 7.2 ms | 82.6 ms |
| steer | 7.3 ms | 81.0 ms |
| write new triple | 7.1 ms | 77.9 ms |
| write duplicate | 0.2 ms | 2.4 ms |
| resident memory | 34 MB | 65 MB |

Reads and writes stay fast because everything happens in memory and the
files are written through. Cold load grows linearly with file count since
every entity is its own JSON file, which is the main limit of this
design. Workspaces in the tens of thousands of nodes stay usable, past
that the startup cost is the thing to watch.

## CLI reference

| Command | Purpose |
|---|---|
| `miea setup` | interactive workspace setup |
| `miea init NAME` | create an empty workspace |
| `miea search QUERY` | find entry points, ranked |
| `miea land REF` | read a node with its signpost |
| `miea steer REF DEST` | move along an edge |
| `miea query-scoped GRAPH QUERY` | search inside one subtree |
| `miea lca A B` | lowest common ancestor of nodes |
| `miea add LABEL` | create a node |
| `miea link A VERB B` | add a named edge |
| `miea wakeup` | session-start snapshot for agent context |
| `miea placement A B` | suggest where a triple belongs |
| `miea verify` | check pending claims against search results |
| `miea provenance` | audit claims without provenance |
| `miea forget REF` | delete a node |

MCP tools mirror these operations under two tiers. Readers get search,
land, steer, scoped queries and LCA. Writers also get add, link, forget,
neighbors, placement, verify and provenance.

## Development

```bash
git clone https://github.com/Kushal2205a/miea-mem.git
cd miea-mem
uv sync
uv run pytest
```

## License

MIT
