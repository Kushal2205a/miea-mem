"""MCP server tests: tiered tool surface via the MCPServer object."""

import json

import pytest

from miea import mcp_server as srv
from miea.core import Memory
from miea.store import Store


@pytest.fixture()
def ws(tmp_path):
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    mem = Memory(str(root))
    mem.create_node("Postgres", "relational db")
    mem.create_node("WAL")
    mem.write_triple("Postgres", "persists_with", "WAL")
    srv._state["mem"] = mem
    srv._state["tier"] = "read"
    return str(root)


async def _call(name: str, arguments: dict) -> str:
    result = await srv.mcp.call_tool(name, arguments)
    return "\n".join(c.text for c in result.content if c.type == "text")


async def test_read_tools_work(ws):
    hits = await _call("search", {"query": "postgres"})
    assert "Postgres" in hits

    landed = await _call("land", {"ref": "Postgres"})
    assert "persists_with" in landed and "WAL" in landed

    steered = await _call("steer",
                          {"ref": "Postgres", "destination": "WAL"})
    assert "--persists_with-->" in steered

    scoped = await _call("query_scoped",
                         {"graph_ref": "T", "query": "wal"})
    assert "[WAL]" in scoped

    ctx = await _call("lca", {"refs": ["Postgres", "WAL"]})
    assert "lca" in json.loads(ctx)


async def test_read_tier_lists_only_read_tools(ws):
    names = {t.name for t in await srv.mcp.list_tools()}
    assert {"search", "land", "steer", "query_scoped", "lca"} == names


async def test_write_tier_adds_tools_and_they_work(ws):
    srv._state["tier"] = "write"
    from miea.mcp_server import _register_write_tools

    _register_write_tools()
    try:
        out = await _call("link", {"source": "Postgres",
                                   "verb": "tuned_via",
                                   "target": "autovacuum"})
        assert "tuned_via" in out
        assert any(n.label == "autovacuum" for n in srv._mem().nodes.values())

        nbrs = await _call("neighbors", {"ref": "Postgres"})
        assert "autovacuum" in nbrs

        added = await _call("add", {"label": "SQLite",
                                    "content": "embedded db"})
        assert "SQLite" in added

        forgot = await _call("forget", {"ref": "SQLite"})
        assert "forgot 1" in forgot

        names = {t.name for t in await srv.mcp.list_tools()}
        assert "link" in names and "forget" in names
    finally:
        # keep other tests isolated: drop write tools again
        for name in ["add", "link", "forget", "neighbors"]:
            srv.mcp.remove_tool(name)


async def test_idempotent_link(ws):
    srv._state["tier"] = "write"
    from miea.mcp_server import _register_write_tools

    _register_write_tools()
    a1 = await _call("link", {"source": "Postgres",
                              "verb": "persists_with", "target": "WAL"})
    a2 = await _call("link", {"source": "Postgres",
                              "verb": "persists_with", "target": "WAL"})
    edges = [e for e in srv._mem().edges.values()
             if e.verb == "persists_with"]
    assert len(edges) == 1
    assert a1 == a2
    for name in ["add", "link", "forget", "neighbors"]:
        srv.mcp.remove_tool(name)
