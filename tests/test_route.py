"""Tests for route: hybrid direction picking over divergence map
entries, cue attribution, the standing-node check, and the ambiguity
flag. Keyword-only (embedder=None) for determinism."""

import pytest

from miea_mem.core import Memory
from miea_mem.store import Store


@pytest.fixture()
def mem(tmp_path) -> Memory:
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    return Memory(str(root), embedder=None)


def _nested_hub(mem: Memory, n: int = 12):
    from miea_mem.model import Graph, new_id

    mem.create_node("Postgres")
    pg = mem._resolve("Postgres")
    child = Graph(id=new_id(), name="PG internals",
                  parent_node_id=pg.id)
    mem.graphs[child.id] = child
    mem.store.save_graph(child)
    pg.child_graph_id = child.id
    mem.store.save_node(pg)
    for i in range(n):
        mem.create_node(f"pgtopic{i}", under_graph=child.id,
                        content=f"topic {i} postgres internals")
        if i % 2 == 0:
            mem.write_triple("Postgres", "uses", f"pgtopic{i}")
    return pg, child


def test_route_attributes_leaf_match_to_branch(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("pgtopic7")              # heat one leaf in the unlinked group
    out = mem.route("Postgres", "pgtopic7")
    top = out["routes"][0]
    assert top["label"] == "Postgres: unlinked"
    assert top["kind"] == "anchor"
    assert top["cue_label"] == "pgtopic7"
    assert out["matched"] is True
    assert out["ambiguous"] is False


def test_route_standing_node_can_be_the_answer(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    out = mem.route("Postgres", "postgres")
    top = out["routes"][0]
    assert top["node_id"] == pg.id
    assert top["kind"] == "self"


def test_route_flags_ambiguity_when_top_two_tie(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("pgtopic2")
    mem.land("pgtopic7")              # one heated leaf per branch
    out = mem.route("Postgres", "topic postgres internals")
    assert out["ambiguous"] is True
    assert len(out["routes"]) >= 2


def test_route_without_map_uses_live_tier(mem: Memory):
    pg, child = _nested_hub(mem)
    out = mem.route("Postgres", "pgtopic3")
    top = out["routes"][0]
    assert top["label"] == "pgtopic3"
    assert top["kind"] == "leaf"
    assert out["matched"] is True


def test_route_still_orders_when_nothing_matches(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    out = mem.route("Postgres", "zzzqqq")
    assert out["matched"] is False
    assert out["ambiguous"] is False
    # nothing filtered: all candidates come back, breadth-ordered
    assert len(out["routes"]) == 3