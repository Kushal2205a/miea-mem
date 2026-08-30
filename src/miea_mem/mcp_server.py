# MCP server. Exposes the memory operations as tools with two access
# tiers. Readers get search, landing, steering and scoped queries.
# Writers additionally get add, link, forget and audits. The tier is
# fixed at startup and never inferred from conversation.

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.mcpserver import MCPServer

from .core import Memory
from .guide import register_guide_resource

mcp = MCPServer(
    "miea",
    instructions=(
        "Graph-based agent memory. Structure and pointers only, you are the "
        "interpreter. Read loop: search then land then slide; route when "
        "you hold a query, steer for single hops, query_scoped on a dead "
        "end. Ranking orders destinations but never hides them. Before "
        "your first write, read memory://guide."
    ),
)

_state: dict[str, Any] = {"tier": "read"}

register_guide_resource(mcp)


def _mem() -> Memory:
    mem: Memory = _state["mem"]
    return mem


# Reader tools

@mcp.tool()
async def search(query: str, limit: int = 5) -> str:
    """Find entry-point memories by keyword. Returns ranked hits (label,
    score) to land on next."""
    hits = _mem().search(query, limit=max(1, min(limit, 20)))
    if not hits:
        return "no matches"
    return "\n".join(
        f"({s:.2f}) [{n.label}] {n.type} -- {n.id}" for n, s in hits
    )


@mcp.tool()
async def land(ref: str, page: int = 0) -> str:
    """Land on a memory node (id or exact label). Returns its content plus a
    signpost: top destinations with the edge verb that reaches each, an
    access score, and epistemic status when noteworthy. Stop here if the
    content answers you; otherwise steer toward the best destination; on a
    dead end use query_scoped. If more destinations exist than shown, land
    again with page=1, 2 and so on to walk the signpost."""
    return _mem().land(ref, page=max(0, page)).render()


@mcp.tool()
async def steer(ref: str, destination: str) -> str:
    """Move from node ref to one of its signpost destinations (label or id).
    Returns the destination's payload plus the sentence-so-far path."""
    return _mem().steer(ref, destination).render()


@mcp.tool()
async def slide(node_ref: str, destination: str, deep: bool = False,
                query: str | None = None) -> str:
    """Ride a branch from node_ref in one pass. Lands at the branch
    entry, or at its cue leaf with deep=true. The payload carries the
    proposition chain (sentence-so-far) and slid-past notes for the
    nodes skipped on the way - check sufficiency on arrival, and land
    on a note's id if it fits better."""
    return _mem().slide(node_ref, destination, deep=deep,
                        query=query).render()


@mcp.tool()
async def route(node_ref: str, query: str, limit: int = 5) -> str:
    """Pick a direction from node_ref's branches by hybrid match
    (keyword + embedding when available + breadth) over its divergence
    map; each anchor inherits its cue leaf's relevance. Returns ranked
    routes; ambiguous=true when the top two tie. Slide into one with
    slide(node_ref, route_label, deep=true)."""
    import json

    return json.dumps(
        _mem().route(node_ref, query, limit=max(1, min(limit, 20))),
        indent=2)


@mcp.tool()
async def query_scoped(graph_ref: str, query: str, limit: int = 5) -> str:
    """Deep dive inside one subtree (graph name or id). The server dives and
    returns matching payloads. Use this instead of exploring when a signpost
    does not show what you need. Ask, do not wander."""
    payloads = _mem().query_scoped(
        graph_ref, query, limit=max(1, min(limit, 20))
    )
    if not payloads:
        return "no matches in that subtree"
    return "\n---\n".join(p.render() for p in payloads)


@mcp.tool()
async def lca(refs: list[str]) -> str:
    """Lowest common ancestor of two or more nodes via parent pointers. The
    minimal coherent context relating them. Pure graph math."""
    import json

    return json.dumps(_mem().lca_context(refs), indent=2)


# Writer tools

def _register_write_tools() -> None:
    @mcp.tool()
    async def add(label: str, content: str = "", type: str = "fact",
                  under_graph: str | None = None,
                  tags: list[str] | None = None) -> str:
        """Create a memory node. Search first to avoid duplicates.

        Choose type deliberately: fact means stable truth, event means
        something that happened at a time, preference means user taste,
        procedure means reusable how-to, claim means checkable world
        statement. Always add category tags, they are how future searches
        find this node."""
        n = _mem().create_node(label, content, type, under_graph)
        if tags:
            n.tags = list(tags)
            from .model import now_iso
            n.updated_at = now_iso()
            _mem()._index_node(n)
            _mem().store.save_node(n)
        return f"created [{n.label}] {n.id}"

    @mcp.tool()
    async def link(source: str, verb: str, target: str,
                   provenance: str | None = None) -> str:
        """Add a named edge SOURCE verb TARGET, for example Postgres
        persists_with WAL. The verb must be a lowercase snake phrase;
        system verbs like corroborated_by are reserved. Missing nodes are
        created. Idempotent. Search before writing to avoid duplicates.
        Pass provenance user_asserts or agent_inferred to record who
        asserts it."""
        try:
            e = _mem().write_triple(source, verb, target, create_missing=True,
                                    provenance=provenance)
        except ValueError as err:
            return f"rejected: {err}"
        extra = f" (+{provenance})" if provenance else ""
        return f"[{source}] --{e.verb}--> [{target}]{extra}"

    @mcp.tool()
    async def forget(ref: str) -> str:
        """Delete a node and its subtree. Only on explicit user request,
        never automatically."""
        removed = _mem().forget(ref)
        return f"forgot {removed} node(s)"

    @mcp.tool()
    async def neighbors(ref: str) -> str:
        """Full local neighborhood of a node, every edge in and out with its
        verb and counterpart. For verification before changes."""
        lines = []
        for n in _mem().neighbors(ref):
            arrow = (f"--{n['verb']}-->" if n["direction"] == "out"
                     else f"<--{n['verb']}--")
            lines.append(f"{arrow} [{n['other']}]")
        return "\n".join(lines) or "no neighbors"

    @mcp.tool()
    async def placement(source: str, target: str) -> str:
        """Where to file a source-target triple: as deep as true, no deeper.
        Returns a structural suggestion."""
        import json

        return json.dumps(_mem().placement_hint(source, target), indent=2)

    @mcp.tool()
    async def verify(limit: int | None = None) -> str:
        """Run the epistemic annotation pass over pending world claims.
        Never deletes anything, only adds status and source edges."""
        import json

        report = _state["verify_pass"]().run(
            limit=max(1, min(limit, 50)) if limit else None)
        return json.dumps(report, indent=2) or "[]"

    @mcp.tool()
    async def provenance() -> str:
        """Audit which claim nodes lack a provenance edge."""
        import json

        return json.dumps(_mem().provenance_report(), indent=2)


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(prog="miea-server")
    parser.add_argument("--root", required=True, help="Workspace directory.")
    parser.add_argument("--tier", choices=["read", "write"], default="read",
                        help="read: signposts only. write: full surface.")
    args = parser.parse_args()

    _state["mem"] = Memory(args.root)
    _state["tier"] = args.tier
    if args.tier == "write":
        from .epistemics import EpistemicPass, NullVerifier

        _register_write_tools()
        # Offline verifier default. A real search backend can be injected
        # here once a provider is configured.
        _state["verify_pass"] = lambda: EpistemicPass(
            _state["mem"], NullVerifier())

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
