# mi∃a — Setup Guide

Graph memory for AI agents. From zero to a working memory wired into your
agent.

## 1. Install

```bash
uv tool install miea-mem
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

Verify:

```bash
miea --help
miea-server --help
```

## 2. Run the setup wizard

```bash
miea setup
```

It asks two questions — workspace location and your name — then:
- creates the workspace + your user anchor node
- detects installed agents (OpenCode, Claude Code, Cursor, Gemini CLI)
- prints a paste-ready MCP config block

Non-interactive equivalent:

```bash
miea setup --root ~/Documents/my_memory --name "Your Name"
```

Setup initializes git history in the workspace automatically, so memory
changes become diffable, revertable commits from day one.

## 3. Use it from the terminal

```bash
alias mie="miea --root ~/Documents/my_memory"

mie add "Postgres" --content "relational database"
mie link Postgres persists_with WAL        # creates missing nodes
mie search relational                      # ranked entry points
mie land Postgres                          # payload + signpost
mie steer Postgres WAL                     # ride one edge
mie wakeup                                 # session-start snapshot (JSON)
mie lca WAL B-trees                        # shared context of two nodes
mie placement Postgres WAL                 # where to file a triple
mie provenance                             # audit unbacked claims
mie verify                                 # epistemic pass (offline)
mie forget WAL                             # deletion (explicit only)
```

## 4. Wire it into an agent (MCP)

Take the JSON block from `miea setup` and paste it into your agent's MCP
config. For OpenCode (`~/.config/opencode/opencode.jsonc`):

```jsonc
{
  "mcp": {
    "my-memory": {
      "type": "local",
      "command": ["/home/USER/.local/share/uv/tools/miea-mem/bin/miea-server",
                  "--root", "~/Documents/my_memory",
                  "--tier", "write"],
      "enabled": true
    }
  }
}
```

Claude Code / Cursor / others use the same server with their own config
shapes.

### Tiers

- `--tier read` → recall only (search/land/steer/query_scoped/lca). Write
  tools don't exist for that agent.
- `--tier write` → full surface (add/link/forget/verify/provenance/placement).

### No AGENTS.md required

The tools teach themselves: descriptions carry usage policy, the server sends
read-loop instructions at handshake, and `memory://guide` holds the full write
discipline. A rules-file snippet is optional polish for power users.

## 5. Custom agents

Any MCP-spec client works — paste the config block into its config. For
agents without hook support, get push-style recall by injecting the snapshot
at session start:

```python
out = subprocess.run(["miea", "--root", WS, "wakeup"], capture_output=True)
prefix = json.loads(out.stdout)["text"]
```

## 6. Multi-agent

Point any number of agents at the same `--root`; mix tiers freely. Writes
from one are visible to the others on their next tool call (fingerprint-based
reload).

## 7. Semantic search

With sentence-transformers installed, `search` is automatically hybrid:
keyword matches fused with paraphrase recall via Reciprocal Rank Fusion.
"what food do I love" finds `[Biryani]` even though "food" appears nowhere in
it. Vectors live in `<workspace>/.index/` (gitignored; delete to rebuild).
