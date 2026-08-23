# agent_mem — Setup Guide

Graph-based memory for AI agents. This guide takes a new user from zero to a
working memory wired into their agent.

## 1. Install

Requires [uv](https://docs.astral.sh/uv/) (or pip) and Python 3.12+.

```bash
# from a clone of this repo:
uv tool install --with sentence-transformers /path/to/agent_mem
```

- `--with sentence-transformers` adds **semantic search** (paraphrase recall,
  ~90MB model auto-downloaded once from HuggingFace on first use, cached
  locally, works offline afterwards). Skip it for a lean keyword-only install.
- After changing the repo, refresh: `uv tool install --force ...`

Verify:

```bash
agent-mem --help
agent-mem-server --help
```

## 2. Create your memory workspace

```bash
agent-mem init "MyMemory" --root ~/Documents/my_memory
cd ~/Documents/my_memory && git init && git add -A && git commit -m "seed"
```

Git is optional but recommended — memory history becomes diffable commits.

## 3. Use it from the terminal

```bash
alias am="agent-mem --root ~/Documents/my_memory"

am add "Postgres" --content "relational database"
am link Postgres persists_with WAL        # creates missing nodes
am search relational                      # hybrid keyword+semantic
am land Postgres                          # payload + signpost
am steer Postgres WAL                     # ride one edge
am lca WAL B-trees                        # shared context of two nodes
am forget WAL                             # deletion (explicit only)
am provenance                             # audit unbacked claims
am verify --verifier null                 # epistemic pass (offline)
am placement Postgres WAL                 # where to file a triple
```

## 4. Wire it into an agent (MCP)

### OpenCode

`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "mcp": {
    "my-memory": {
      "type": "local",
      "command": ["/home/YOU/.local/bin/agent-mem-server",
                  "--root", "/home/YOU/Documents/my_memory",
                  "--tier", "write"],
      "enabled": true
    }
  }
}
```

Plus `~/.config/opencode/AGENTS.md` (see `examples/AGENTS.md`) so the agent
knows *when* to use memory, not just how.

### Claude Code / other MCP clients

Same server, their config format:

```json
{
  "mcpServers": {
    "my-memory": {
      "command": "/home/YOU/.local/bin/agent-mem-server",
      "args": ["--root", "/home/YOU/Documents/my_memory", "--tier", "write"]
    }
  }
}
```

### Tiers

- `--tier read` → agent can only recall (search/land/steer/query_scoped/lca).
  Write tools do not exist for it — cannot be talked into mutating memory.
- `--tier write` → full surface (add/link/forget/verify/provenance/placement).
  Give write only to agents you trust to learn.

## 5. What the agent gets

- **Server instructions**: the read loop (search → land → steer; stop when
  satisfied; query_scoped on dead ends).
- **`memory://guide` resource**: full write policy — salience test, triple
  grammar, provenance discipline, what not to store.
- **`memory://reflect` prompt**: end-of-session distillation ritual.
- **Guardrails**: malformed verbs rejected with teaching messages; system
  epistemic verbs reserved; duplicates idempotent.
- **Multi-agent safe**: long-lived servers detect external writes via
  fingerprint and reload; concurrent CLI + MCP writers coexist.

## 6. Multi-agent

Any number of agents can share one workspace: give each the same `--root`.
Mix tiers freely (a read-only researcher + a write-enabled assistant).
Writes from one are visible to the others on their next tool call.

## 7. Semantic search (optional but recommended)

With sentence-transformers installed (step 1), search automatically becomes
hybrid: exact keyword matches (FTS) fused with paraphrase recall (embeddings)
via Reciprocal Rank Fusion. "what food do I love" finds `[Biryani]` even
though the word "food" appears nowhere in it.

Vectors live in `<workspace>/.index/` (gitignored, rebuildable). To force a
rebuild: delete `.index/` — it regenerates on next load.

## 8. Verify it all works

```bash
am search "anything you stored"     # should hit
```

Then ask your agent: "Do you remember what food I love?" — it should search
memory first, not ask you.
