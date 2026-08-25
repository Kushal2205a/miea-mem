# mi∫a

Graph memory for AI agents. Nodes hold facts. Edges name the relationship.
Paths through the graph read as sentences: Kushal studied_at Woxsen
University.

```
        search("university")
             |
             v
   [* Kushal]
     |  destinations:
     |  [Biryani]          --likes-->
     |  [Blade Runner]     --likes-->      read the signpost,
     |  [Woxsen]           --studied_at--> pick a verb,
     |  [Capstone story]   --experienced-> steer once.
     |
     +-------- steer ---------->  [* Woxsen University]
                                       content found, answer ready
```

## Install

```bash
uv tool install miea-mem

# with semantic search (paraphrase recall, downloads a local model):
uv tool install "miea-mem[semantic]"
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

Tell your agent things and it stores them:

> remember that I like biryani

Later, ask it questions:

> what food do I like?

Under the hood the agent runs `search` to find an entry node, reads its
content and signpost with `land`, then follows named edges with `steer`
until it has the answer.

You can also drive it from the terminal:

```bash
miea --root ~/Documents/my-memory add Postgres --content "relational database"
miea --root ~/Documents/my-memory link Postgres persists_with WAL
miea --root ~/Documents/my-memory land Postgres
miea --root ~/Documents/my_memory search relational
```

## How it works

Storage is one JSON file per node, edge and graph. On startup these load
into flat dicts plus three derived indexes: word counts for keyword
search, adjacency lists for traversal, and a containment map for nesting.

Reading a node returns its content and a signpost of destinations. Each
destination carries the verb of the edge that reaches it and an access
score. Scores order the list. Nothing is hidden, paging and scoped
queries reach everything stored.

Writing takes noun verb noun triples. Duplicate triples do nothing.
Verbs must be lowercase snake phrases. A small set of epistemic verbs is
reserved so stored claims can only be corroborated or contradicted by the
verify pass, which checks pending claims against search results when you
run it.

## Design rules

- The consuming agent does all interpretation. This package stores
  structure and pointers, not summaries.
- Workspace files are plain JSON, readable and editable by hand. They are
  the only source of truth. Derived indexes rebuild from them at any time.
- Access counts rank results but never filter them. Every memory stays
  reachable, low rank only costs latency.
- Deletion is explicit. Nothing decays or disappears on its own.

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
