"""Tests for the divergence map data model: JSON round-trip, tolerance of
old files without the field, and the pointer-only contract (labels and
ids, never generated text)."""

import json

import pytest

from miea_mem.core import Memory
from miea_mem.model import (
    DivergenceEntry,
    Node,
    node_from_dict,
    node_to_dict,
)
from miea_mem.store import Store


@pytest.fixture()
def mem(tmp_path) -> Memory:
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    return Memory(str(root))


def test_divergence_entry_round_trips_through_json():
    n = Node(id="n1", label="Food", type="anchor")
    n.divergence_map = [
        DivergenceEntry(node_id="a1", label="Japanese cuisine",
                        kind="anchor", cue_leaf_id="s1", cue_label="Sushi"),
        DivergenceEntry(node_id="l1", label="Biryani", kind="leaf"),
    ]
    restored = node_from_dict(json.loads(json.dumps(node_to_dict(n))))
    assert restored.divergence_map == n.divergence_map
    assert restored.divergence_map[0].kind == "anchor"
    assert restored.divergence_map[1].cue_leaf_id is None


def test_node_without_map_field_loads_empty_and_writes_compact():
    # an old workspace file without divergenceMap must load with an
    # empty map, and a node with no map must not serialize the key at
    # all so existing files keep their byte shape
    old = {
        "schemaVersion": 1, "id": "n2", "type": "fact", "label": "WAL",
        "tags": [], "content": "write ahead log",
        "epistemicStatus": "unverifiable",
        "breadth": {"accessCount": 0, "traversalCount": 0,
                    "lastAccessed": None},
        "createdAt": "2026-01-01T00:00:00+00:00",
        "updatedAt": "2026-01-01T00:00:00+00:00",
    }
    n = node_from_dict(old)
    assert n.divergence_map == []
    assert "divergenceMap" not in node_to_dict(Node(id="n3", label="x"))


def test_map_stores_pointers_not_content():
    n = Node(id="n4", label="Food")
    n.divergence_map = [
        DivergenceEntry(node_id="a1", label="Japanese cuisine",
                        kind="anchor", cue_leaf_id="s1", cue_label="Sushi"),
    ]
    d = node_to_dict(n)
    # entry dicts carry identity and routing only, no free text beyond
    # the labels themselves
    assert all(set(e) <= {"nodeId", "label", "kind",
                          "cueLeafId", "cueLabel"}
               for e in d["divergenceMap"])


# -- map builder + lazy regeneration -------------------------------------------


def _nested_hub(mem: Memory, n: int = 12):
    # Postgres owns child graph "PG internals" with n plain members,
    # even indices linked with 'uses' so a split forms two anchor groups.
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
        node = mem.create_node(f"pgtopic{i}", under_graph=child.id)
        if i == 0:
            node.content = "secret sauce vacuum tuning"
            mem.store.save_node(node)
        if i % 2 == 0:
            mem.write_triple("Postgres", "uses", f"pgtopic{i}")
    return pg, child


def test_fork_map_built_on_access_after_split(mem: Memory):
    pg, child = _nested_hub(mem)
    created = mem.split_if_overloaded(pg.id, cap=5)
    assert len(created) >= 2
    mem.land("Postgres", mark_access=False)
    entries = mem.nodes[pg.id].divergence_map
    assert [e.kind for e in entries] == ["anchor", "anchor"]
    assert [e.label for e in entries] == sorted(e.label for e in entries)
    for e in entries:
        sub = mem.graphs[mem.nodes[e.node_id].child_graph_id]
        assert e.cue_leaf_id in sub.node_ids
        assert e.cue_label == mem.nodes[e.cue_leaf_id].label
    # anchors over plain leaves hold no stored map: their entries equal
    # the live sibling view, so nothing is warranted
    mem.land(mem.nodes[created[0]].label, mark_access=False)
    assert mem.nodes[created[0]].divergence_map == []


def test_singleton_branch_stores_leaf_entry(mem: Memory):
    pg, child = _nested_hub(mem, n=3)
    created = mem.split_if_overloaded(pg.id, cap=2)
    assert len(created) == 1        # the unlinked singleton is not wrapped
    mem.land("Postgres", mark_access=False)
    entries = mem.nodes[pg.id].divergence_map
    assert {e.kind for e in entries} == {"anchor", "leaf"}
    leaf_entry = next(e for e in entries if e.kind == "leaf")
    assert leaf_entry.node_id == mem._resolve("pgtopic1").id
    assert leaf_entry.cue_leaf_id is None
    anchor_entry = next(e for e in entries if e.kind == "anchor")
    assert anchor_entry.cue_leaf_id in {
        mem._resolve("pgtopic0").id, mem._resolve("pgtopic2").id}


def test_write_dirties_fork_until_accessed(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("Postgres", mark_access=False)
    built = mem.nodes[pg.id].divergence_map
    assert len(built) == 2
    mem.create_node("pgextra", under_graph=child.id)
    # stale until touched: the stored map does not know pgextra yet
    assert mem.nodes[pg.id].divergence_map == built
    assert pg.id in mem._dirty_maps
    mem.land("Postgres", mark_access=False)
    refreshed = mem.nodes[pg.id].divergence_map
    assert len(refreshed) == 3
    assert "pgextra" in (e.label for e in refreshed)
    assert pg.id not in mem._dirty_maps


def test_forget_removes_map_entry(mem: Memory):
    pg, child = _nested_hub(mem)
    created = mem.split_if_overloaded(pg.id, cap=5)
    mem.land("Postgres", mark_access=False)
    assert len(mem.nodes[pg.id].divergence_map) == 2
    mem.forget(created[0])
    mem.land("Postgres", mark_access=False)
    entries = mem.nodes[pg.id].divergence_map
    assert len(entries) == 1
    assert entries[0].node_id == created[1]


def test_map_persists_to_file_and_reloads(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("Postgres", mark_access=False)
    built = mem.nodes[pg.id].divergence_map
    data = json.loads(
        (mem.store.root / "nodes" / f"{pg.id}.json").read_text())
    # pointer-only on disk: member content never leaks into the map
    assert all("secret sauce" not in json.dumps(e)
               for e in data["divergenceMap"])
    fresh = Memory(str(mem.store.root))
    assert fresh.nodes[pg.id].divergence_map == built


def test_reindex_rebuilds_maps_lazily(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("Postgres", mark_access=False)
    built = mem.nodes[pg.id].divergence_map
    mem.nodes[pg.id].divergence_map = []      # simulate drift
    mem.reindex()
    assert pg.id in mem._dirty_maps
    mem.land("Postgres", mark_access=False)
    assert mem.nodes[pg.id].divergence_map == built
    assert pg.id not in mem._dirty_maps


def test_unsplit_tier_stays_map_free(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.land("Postgres", mark_access=False)
    # plain topics only, no anchors: nothing worth storing
    assert mem.nodes[pg.id].divergence_map == []


def test_cue_updates_when_leaf_overtakes_it(mem: Memory):
    # the recency tiebreaker works live, without a structural write
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    mem.land("Postgres", mark_access=False)
    fork = mem.nodes[pg.id]
    unlinked = next(e for e in fork.divergence_map
                    if "unlinked" in e.label)
    old_cue = unlinked.cue_leaf_id
    assert old_cue is not None
    # heat a DIFFERENT leaf in the same branch until it beats the cue
    branch_members = [
        nid for nid in mem.nodes
        if nid != old_cue and _is_child_of(mem, nid, unlinked.node_id)]
    target = next(nid for nid in branch_members if nid in mem.nodes)
    for _ in range(3):
        mem.land(mem.nodes[target].label)
    refreshed = mem.nodes[pg.id]
    new_entry = next(e for e in refreshed.divergence_map
                     if "unlinked" in e.label)
    assert new_entry.cue_leaf_id == target
    # route now attributes a leaf-specific query to the right branch
    out = mem.route("Postgres", mem.nodes[target].label)
    assert out["routes"][0]["node_id"] == unlinked.node_id
    assert out["routes"][0]["cue_id"] == target


def _is_child_of(mem: Memory, nid: str, anchor_id: str) -> bool:
    # nid lives inside the anchor's owned sub-graph
    anchor = mem.nodes.get(anchor_id)
    g = mem.graphs.get(anchor.child_graph_id) if anchor else None
    return bool(g and nid in g.node_ids)


def test_cue_updates_for_root_fork_tier_mates(mem: Memory):
    # root-level fork: anchors are tier-mates of the fork node, their
    # parent chain stops at the root graph; the cue still refreshes live
    mem.create_node("User")
    user = mem._resolve("User")
    for i in range(12):
        mem.create_node(f"u{i}", content=f"topic {i} postgres vacuum")
        if i % 2 == 0:
            mem.write_triple("User", "uses", f"u{i}")
    mem.split_if_overloaded(user.id, cap=5)
    mem.land("User", mark_access=False)
    fork = mem.nodes[user.id]
    unlinked = next(e for e in fork.divergence_map
                    if "unlinked" in e.label)
    old_cue = unlinked.cue_leaf_id
    target = next(nid for nid in mem.nodes
                  if nid != old_cue and mem.nodes[nid].label != "User"
                  and _is_child_of(mem, nid, unlinked.node_id))
    for _ in range(3):
        mem.land(mem.nodes[target].label)
    refreshed = next(e for e in mem.nodes[user.id].divergence_map
                     if "unlinked" in e.label)
    assert refreshed.cue_leaf_id == target


def _is_child_of(mem: Memory, nid: str, anchor_id: str) -> bool:
    # nid lives inside the anchor's owned sub-graph
    anchor = mem.nodes.get(anchor_id)
    g = mem.graphs.get(anchor.child_graph_id) if anchor else None
    return bool(g and nid in g.node_ids)


# -- signpost rendering through the map ----------------------------------------


def test_signpost_serves_map_entries_with_cues(mem: Memory):
    pg, child = _nested_hub(mem)
    mem.split_if_overloaded(pg.id, cap=5)
    dests = mem._all_destinations(mem._resolve("Postgres"))
    # edge rows (uses -> topics) plus one map row per anchor branch
    child_rows = [d for d in dests if d.direction == "child"]
    assert len(child_rows) == 2
    assert all(d.cue_label for d in child_rows)
    assert all(d.cue_node_id in mem.graphs[
        mem.nodes[d.node_id].child_graph_id].node_ids
        for d in child_rows)
    assert all("cue: [" in d.render() for d in child_rows)
    # edge rows stay verb-mediated and cue-free
    edge_rows = [d for d in dests if d.direction == "out"]
    assert len(edge_rows) == 6
    assert all(d.cue_label is None and d.verb == "uses" for d in edge_rows)
    # the payload pages the full set: 6 edge rows + 2 branch rows
    p0 = mem.land("Postgres", mark_access=False)
    assert p0.total_destinations == 8


def test_signpost_without_map_keeps_live_siblings(mem: Memory):
    pg, child = _nested_hub(mem)
    dests = mem._all_destinations(mem._resolve("Postgres"))
    child_rows = [d for d in dests if d.direction == "child"]
    # the six uses-linked topics appear as edge rows and the edge view
    # wins dedup, so the live sibling tier shows the six unlinked ones
    assert len(child_rows) == 6
    assert all(d.cue_label is None for d in child_rows)


def test_mixed_map_rows_in_signpost(mem: Memory):
    pg, child = _nested_hub(mem, n=3)
    mem.split_if_overloaded(pg.id, cap=2)
    mem.land("Postgres", mark_access=False)
    dests = mem._all_destinations(mem._resolve("Postgres"))
    child_rows = [d for d in dests if d.direction == "child"]
    assert len(child_rows) == 2           # anchor branch + singleton leaf
    anchor_row = next(d for d in child_rows if d.cue_label)
    leaf_row = next(d for d in child_rows if not d.cue_label)
    assert leaf_row.node_id == mem._resolve("pgtopic1").id
    assert leaf_row.label == "pgtopic1"