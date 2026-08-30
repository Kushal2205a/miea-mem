"""End-to-end workflow test: search -> land -> route -> slide -> split.
Pins the designed read loop as one runnable chain so the workflow stays
faithful to the divergence-map read path."""

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
                        content=f"topic {i} postgres vacuum internals")
        if i % 2 == 0:
            mem.write_triple("Postgres", "uses", f"pgtopic{i}")
    return pg, child


def test_workflow_search_land_route_slide(mem: Memory):
    _nested_hub(mem)
    mem.land("pgtopic7")   # the target gets touched before the split

    # 1. search finds an entry node
    hits = [n.label for n, _ in mem.search("postgres")]
    assert "Postgres" in hits

    # 2. land shows the fork
    mem.land("Postgres", mark_access=False)

    # 3. split: fanout pressure promotes branches
    pg = mem._resolve("Postgres")
    created = mem.split_if_overloaded(pg.id, cap=5)
    assert len(created) >= 2
    mem.land("Postgres", mark_access=False)
    entries = mem.nodes[pg.id].divergence_map
    assert len([e for e in entries if e.kind == "anchor"]) >= 2

    # 4. route picks the branch holding the target
    out = mem.route("Postgres", "pgtopic7")
    top = out["routes"][0]
    assert top["kind"] == "anchor"
    assert top["cue_label"] == "pgtopic7"
    assert out["matched"] is True

    # 5. slide deep rides straight to the answer in one pass
    p = mem.slide("Postgres", top["label"], deep=True)
    assert p.node.label == "pgtopic7"
    assert p.path_so_far == ["[Postgres]", ">", f"[{top['label']}]",
                             ">", "[pgtopic7]"]
    assert p.node.content == "topic 7 postgres vacuum internals"

    # 6. suggest_split reports no tie on the unambiguous route
    diag = mem.suggest_split("Postgres", "pgtopic7")
    assert diag["matched"] is True
    assert diag["ambiguous"] is False


def test_workflow_leaf_specific_query_attributes_to_branch(mem: Memory):
    _nested_hub(mem)
    mem.land("pgtopic7")
    pg = mem._resolve("Postgres")
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("Postgres", mark_access=False)
    out = mem.route("Postgres", "pgtopic7")
    assert out["routes"][0]["cue_label"] == "pgtopic7"
    assert out["routes"][0]["kind"] == "anchor"