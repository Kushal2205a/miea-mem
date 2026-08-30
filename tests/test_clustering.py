"""Tests for clustering strategies and the split suggestion diagnostic:
semantic grouping fixes the same-verb-strangers / similar-different-verbs
failure modes of the verb signature, and suggest_split previews the cut
without mutating anything."""

import pytest

from miea_mem.core import Memory
from miea_mem.store import Store


class StubEmbedder:
    # Same deterministic toy embedder as the semantic tests: seed words
    # map to fixed axes so tests can control similarity.
    dim = 8
    AXES = {
        "food": 0, "rice": 0, "cuisine": 0,
        "db": 1, "postgres": 1, "sql": 1, "database": 1,
    }

    def embed(self, texts):
        import math

        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in t.lower().split():
                if tok in self.AXES:
                    v[self.AXES[tok]] = 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture()
def mem(tmp_path) -> Memory:
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    return Memory(str(root), embedder=StubEmbedder())


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


def _verb_twin_hub(mem: Memory):
    # Four members under a hub: sushi/sashimi are content-twins with
    # different verbs; biryani shares sushi's verb but is also food;
    # wal is db-family with sushi's verb. The verb signature cuts the
    # wrong way on both counts; the semantic cut goes along content.
    hub = mem.create_node("hub")
    for label, verb, content in [
        ("sushi", "likes", "japanese food rice cuisine"),
        ("sashimi", "prefers", "japanese food rice cuisine"),
        ("biryani", "likes", "indian food rice cuisine"),
        ("wal", "likes", "db postgres sql database"),
    ]:
        mem.create_node(label, content=content)
        mem.write_triple("hub", verb, label)
    return hub


def _sub_members(mem: Memory, created: list[str]) -> set:
    labels = set()
    for gid in created:
        sub = mem.graphs[mem.nodes[gid].child_graph_id]
        labels |= {mem.nodes[nid].label for nid in sub.node_ids}
    return labels


def test_verb_strategy_groups_same_verb_strangers(mem: Memory):
    hub = _verb_twin_hub(mem)
    created = mem.split_if_overloaded(hub.id, cap=2)
    members = _sub_members(mem, created)
    # the failure mode on record: wal (db) lands with the food nodes
    assert {"sushi", "biryani", "wal"} <= members


def test_semantic_strategy_cuts_along_content(mem: Memory):
    hub = _verb_twin_hub(mem)
    created = mem.split_if_overloaded(hub.id, cap=2, strategy="semantic")
    members = _sub_members(mem, created)
    # food stays together, db stays out: same-verb strangers separated,
    # similar-different-verbs grouped
    assert {"sushi", "sashimi", "biryani"} <= members
    assert "wal" not in members


def test_semantic_falls_back_to_verbs_without_embedder(tmp_path):
    Store(tmp_path / "ws2").init_workspace("T")
    mem = Memory(str(tmp_path / "ws2"), embedder=None)
    mem.create_node("hub9")
    for i in range(6):
        mem.create_node(f"t{i}")
        mem.write_triple("hub9", "uses", f"t{i}")
    for i in range(6, 12):
        mem.create_node(f"t{i}")
    created = mem.split_if_overloaded(mem._resolve("hub9").id, cap=5,
                                      strategy="semantic")
    assert len(created) == 2
    assert sorted(mem.nodes[g].label for g in created) == \
        ["hub9: unlinked", "hub9: uses"]


def test_suggest_split_reports_tie_and_preview(mem: Memory):
    # the diagnostic runs BEFORE the split: that is its place in the
    # workflow - read the tie, preview the cut, then split explicitly
    pg, child = _nested_hub(mem)
    mem.land("pgtopic2")
    mem.land("pgtopic7")
    out = mem.suggest_split("Postgres", "topic postgres internals")
    assert out["matched"] is True
    assert out["ambiguous"] is True
    assert len(out["tied_routes"]) >= 2
    # every topic carries the postgres axis, so the preview sees one
    # content cluster spanning all twelve members
    assert out["semantic_groups"] is not None
    assert len(out["semantic_groups"]) == 1
    assert sum(len(g["labels"]) for g in out["semantic_groups"]) == 12


def test_suggest_split_without_signal_is_quiet(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    out = mem.suggest_split("Postgres", "zzzqqq")
    assert out["matched"] is False
    assert out["tied_routes"] == []