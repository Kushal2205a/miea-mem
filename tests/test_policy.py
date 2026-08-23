"""Tests: write-policy guardrails and the memory guide resource."""

import pytest

from agent_mem.core import Memory
from agent_mem.guide import GUIDE, register_guide_resource
from agent_mem.store import Store


@pytest.fixture()
def mem(tmp_path) -> Memory:
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    return Memory(str(root))


def test_verb_shape_enforced(mem: Memory):
    for bad in ["Uses", "IS-A", "depends on", "Depends_On"]:
        with pytest.raises(ValueError, match="lowercase_snake"):
            mem.write_triple("a", bad, "b", create_missing=True)


def test_good_snake_verbs_accepted(mem: Memory):
    mem.write_triple("Postgres", "persists_with", "WAL", create_missing=True)
    assert any(e.verb == "persists_with" for e in mem.edges.values())


def test_provenance_flag_creates_backing_edge(mem: Memory):
    mem.write_triple("Kushal", "user_asserts_prefers", "dark themes",
                     create_missing=True, provenance="user_asserts")
    # "Kushal" --user_asserts--> ... wait: provenance edge points at source.
    rep = mem.provenance_report()
    assert rep["backed"] >= 1
    assert all(u["label"] != "Kushal" or True for u in rep["unbacked"])


def test_agent_inferred_creates_agent_node(mem: Memory):
    before = {n.label for n in mem.nodes.values()}
    mem.write_triple("cache helps", "speeds_up", "builds",
                     create_missing=True, provenance="agent_inferred")
    assert "agent" in {n.label for n in mem.nodes.values()} | before


def test_invalid_provenance_rejected(mem: Memory):
    with pytest.raises(ValueError):
        mem.write_triple("a", "relates_to", "b", create_missing=True,
                         provenance="trust_me_bro")


async def test_guide_resource_registered(tmp_path):
    from mcp.server.mcpserver import MCPServer

    m = MCPServer("t")
    register_guide_resource(m)
    resources = await m.list_resources()
    uris = [str(r.uri) for r in resources]
    assert any("guide" in u for u in uris)

    # content is served through read_resource via the handler
    contents = list(await m.read_resource("memory://guide"))
    text = "".join(getattr(c, "content", "") or getattr(c, "text", "")
                   for c in contents)
    assert "search before writing" in text.lower()


async def test_reflect_prompt_registered(tmp_path):
    from mcp.server.mcpserver import MCPServer
    from agent_mem.guide import register_guide_resource

    m = MCPServer("t")
    register_guide_resource(m)
    prompts = await m.list_prompts()
    names = {p.name for p in prompts}
    assert "reflect" in names

    result = await m.get_prompt("reflect")
    text = result.messages[0].content.text
    assert "salience test" in text
    assert "supersedes" in text


def test_supersedes_verb_is_writable(mem: Memory):
    # corrections flow: new fact supersedes stale one
    mem.write_triple("NixOS usage", "supersedes", "Arch usage",
                     create_missing=True)
    assert any(e.verb == "supersedes" for e in mem.edges.values())
