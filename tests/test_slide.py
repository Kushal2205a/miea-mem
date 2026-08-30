"""Tests for the slide primitive: one-pass descent, the proposition
chain, transit notes for skipped nodes, and breadth honesty."""

import pytest

from miea_mem.core import Memory
from miea_mem.store import Store


@pytest.fixture()
def mem(tmp_path) -> Memory:
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    return Memory(str(root))


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


def test_slide_lands_at_branch_entry_without_transit(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    anchor = next(n for n in mem.nodes.values() if n.type == "anchor")
    p = mem.slide("Postgres", anchor.label, mark_access=False)
    assert p.node.id == anchor.id
    assert p.path_so_far == ["[Postgres]", ">", f"[{anchor.label}]"]
    assert p.transit_notes == []
    # breadth honesty: both ridden nodes count the traversal, and the
    # landing gets no access bump when mark_access is off
    assert pg.breadth.traversal_count == 1
    assert anchor.breadth.traversal_count == 1
    assert anchor.breadth.access_count == 0


def test_deep_slide_rides_to_cue_with_transit_notes(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("pgtopic7")              # heat one leaf in the unlinked group
    anchor = next(n for n in mem.nodes.values() if n.type == "anchor"
                  and "unlinked" in n.label)
    p = mem.slide("Postgres", anchor.label, deep=True, mark_access=False)
    assert p.node.label == "pgtopic7"     # cue: hottest leaf in the branch
    assert p.path_so_far == ["[Postgres]", ">",
                             f"[{anchor.label}]", ">", "[pgtopic7]"]
    assert len(p.transit_notes) == 1
    assert p.transit_notes[0]["node_id"] == anchor.id
    # split-created anchors are structural and content-free, so the
    # note documents the skipped route point without a snippet
    assert p.transit_notes[0]["snippet"] == ""
    assert "slid past" in p.render()
    # the skipped anchor was ridden, so it counts a traversal too
    assert anchor.breadth.traversal_count == 1
    assert pg.breadth.traversal_count == 1
    assert p.node.breadth.traversal_count == 1


def test_slide_query_ranks_transit_notes(mem: Memory):
    # a deeper chain, Postgres -> A -> B -> leaf, gives two transit nodes
    from miea_mem.model import Graph, new_id

    mem.create_node("Postgres")
    pg = mem._resolve("Postgres")
    ga = Graph(id=new_id(), name="A", parent_node_id=pg.id)
    mem.graphs[ga.id] = ga
    mem.store.save_graph(ga)
    pg.child_graph_id = ga.id
    mem.store.save_node(pg)
    a = mem.create_node("Branch A", type="anchor", under_graph=ga.id,
                        content="general branch")
    sub = Graph(id=new_id(), name="A inner", parent_node_id=a.id)
    mem.graphs[sub.id] = sub
    mem.store.save_graph(sub)
    a.child_graph_id = sub.id
    mem.store.save_node(a)
    b = mem.create_node("Branch B", type="anchor", under_graph=sub.id,
                        content="vacuum tuning details")
    deep = Graph(id=new_id(), name="B inner", parent_node_id=b.id)
    mem.graphs[deep.id] = deep
    mem.store.save_graph(deep)
    b.child_graph_id = deep.id
    mem.store.save_node(b)
    leaf = mem.create_node("vacuum notes", under_graph=deep.id,
                           content="vacuum tuning specifics")

    p = mem.slide("Postgres", "Branch A", deep=True, query="vacuum",
                  mark_access=False)
    assert p.node.id == leaf.id       # A's cue is the hottest subtree node
    labels = [n["label"] for n in p.transit_notes]
    assert labels == ["Branch B", "Branch A"]   # query-relevant first
    assert p.transit_notes[0]["overlap"] >= 1
    assert p.transit_notes[1]["overlap"] == 0
    assert p.path_so_far == ["[Postgres]", ">", "[Branch A]", ">",
                             "[Branch B]", ">", "[vacuum notes]"]


def test_slide_rejects_unrelated_nodes(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    # unrelated node: not even a destination of the standing node
    with pytest.raises(LookupError, match="no destination matching"):
        mem.slide("pgtopic0", "pgtopic1")     # different branches
    # edge hop up: it IS a destination (via the uses edge), but not
    # beneath or beside the standing node, so the slide refuses and
    # points at steer
    with pytest.raises(LookupError, match="not reachable"):
        mem.slide("pgtopic0", "Postgres")     # steer instead


def test_slide_deep_without_cue_lands_at_entry(mem: Memory):
    pg, child = _nested_hub(mem, n=3)
    mem.split_if_overloaded(pg.id, cap=2)
    # pgtopic1 is a singleton leaf branch: deep has nowhere further to go
    p = mem.slide("Postgres", "pgtopic1", deep=True, mark_access=False)
    assert p.node.label == "pgtopic1"
    assert p.transit_notes == []
    assert p.path_so_far == ["[Postgres]", ">", "[pgtopic1]"]