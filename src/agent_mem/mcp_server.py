"""MCP server: exposes agent_mem's tiered tool surface over the MCP protocol.

READ tier (default): search, land, steer, query_scoped, lca
WRITE tier: everything above + add, link, forget, neighbors

Tier is fixed at startup — never inferred from conversation:
    agent-mem-server --root DIR [--tier read|write]
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.mcpserver import MCPServer

from .core import Memory

mcp = MCPServer(
    "agent-mem",
    instructions=(
        "Graph-based agent memory. Structure and pointers only — you are the "
        "interpreter. Read loop: search → land → steer. Stop when a node's "
        "content satisfies you; steer along signpost verbs when routed; use "
        "query_scoped on a dead end. Ranking orders destinations but never "
        "hides them."
    ),
)

_state: dict[str, Any] = {"tier": "read"}


def _mem() -> Memory:
    mem: Memory = _state["mem"]
    return mem


# ---------------------------------------------------------------------------
# READ tier
# ---------------------------------------------------------------------------


@mcp.tool()
async def search(query: str, limit: int = 5) -> str:
    """Find entry-point memories by keyword. Returns ranked hits (label,
    score) to land on next."""
    hits = _mem().search(query, limit=max(1, min(limit, 20)))
    if not hits:
        return "no matches"
    return "\n".join(
        f"({s:.2f}) [{n.label}] {n.type} — {n.id}" for n, s in hits
    )


@mcp.tool()
async def land(ref: str) -> str:
    """Land on a memory node (id or exact label). Returns its content plus a
    signpost: top-k destinations, each with the named edge verb that reaches
    it and a breadth score. Stop here if the content answers you; otherwise
    steer toward the best destination; on a dead end use query_scoped."""
    return _mem().land(ref).render()


@mcp.tool()
async def steer(ref: str, destination: str) -> str:
    """Ride one branch of the slide: move from node `ref` to one of its
    signpost destinations (label or id). Returns the destination's payload
    plus the sentence-so-far path."""
    return _mem().steer(ref, destination).render()


@mcp.tool()
async def query_scoped(graph_ref: str, query: str, limit: int = 5) -> str:
    """Mediated deep dive inside one subtree (graph name or id). The server
    dives and returns matching payloads. Use this instead of exploring when a
    signpost doesn't show what you need — ask, don't wander."""
    payloads = _mem().query_scoped(
        graph_ref, query, limit=max(1, min(limit, 20))
    )
    if not payloads:
        return "no matches in that subtree"
    return "\n---\n".join(p.render() for p in payloads)


@mcp.tool()
async def lca(refs: list[str]) -> str:
    """Lowest common ancestor of 2+ nodes via parent pointers — the minimal
    coherent context relating them. Pure graph math."""
    import json

    return json.dumps(_mem().lca_context(refs), indent=2)


# ---------------------------------------------------------------------------
# WRITE tier (registered only when --tier write)
# ---------------------------------------------------------------------------


def _register_write_tools() -> None:
    @mcp.tool()
    async def add(label: str, content: str = "", type: str = "fact",
                 under_graph: str | None = None) -> str:
        """(write tier) Create a memory node. Search first to avoid duplicates."""
        n = _mem().create_node(label, content, type, under_graph)
        return f"created [{n.label}] {n.id}"

    @mcp.tool()
    async def link(source: str, verb: str, target: str) -> str:
        """(write tier) Add a named edge SOURCE --VERB--> TARGET, e.g.
        'Postgres persists_with WAL'. Missing nodes are created. Idempotent.
        Search before writing to avoid duplicate propositions."""
        e = _mem().write_triple(source, verb, target, create_missing=True)
        return f"[{source}] --{e.verb}--> [{target}]"

    @mcp.tool()
    async def forget(ref: str) -> str:
        """(write tier) Explicitly delete a node and its subtree. Only ever
        called on direct user request — never automatically."""
        removed = _mem().forget(ref)
        return f"forgot {removed} node(s)"

    @mcp.tool()
    async def neighbors(ref: str) -> str:
        """(write tier) Full local neighborhood of a node — every in/out edge
        with verb and counterpart. For verification before surgery."""
        lines = []
        for n in _mem().neighbors(ref):
            arrow = (f"--{n['verb']}-->" if n["direction"] == "out"
                     else f"<--{n['verb']}--")
            lines.append(f"{arrow} [{n['other']}]")
        return "\n".join(lines) or "no neighbors"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-mem-server")
    parser.add_argument("--root", required=True, help="Workspace directory.")
    parser.add_argument("--tier", choices=["read", "write"], default="read",
                        help="read: signposts only. write: full surface.")
    args = parser.parse_args()

    _state["mem"] = Memory(args.root)
    _state["tier"] = args.tier
    if args.tier == "write":
        _register_write_tools()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
