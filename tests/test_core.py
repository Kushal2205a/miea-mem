"""Core flow tests: model, read loop, write tier, LCA."""

from pathlib import Path

import pytest

from miea_mem.core import Memory
from miea_mem.store import Store


@pytest.fixture()
def ws(tmp_path: Path) -> str:
    root = tmp_path / "ws"
    Store(root).init_workspace("Test")
    return str(root)


@pytest.fixture()
def mem(ws: str) -> Memory:
    m = Memory(ws)
    # Build a small domain:
    #   [Databases] --depends_on--> [B-trees]
    #   [Databases] --degrades_without--> [Vacuum tuning]
    #   nested under Databases: [Postgres], [MySQL]
    dbs = m.create_node("Databases", "storage systems", type="fact")
    bt = m.create_node("B-trees")
    vac = m.create_node("Vacuum tuning")
    pg = m.create_node("Postgres")
    my = m.create_node("MySQL")
    m.write_triple("Databases", "depends_on", "B-trees")
    m.write_triple("Databases", "degrades_without", "Vacuum tuning")
    # put pg/my inside a child graph owned by a node? keep flat siblings here
    del dbs, pg, my
    return m


def test_init_creates_files(ws: str):
    root = Path(ws)
    assert (root / "manifest.json").exists()
    assert list((root / "nodes").glob("*.json")) == []
    assert len(list((root / "graphs").glob("*.json"))) == 1


def test_search_finds_by_tokens(mem: Memory):
    hits = mem.search("storage systems")
    assert hits and hits[0][0].label == "Databases"


def test_land_returns_signpost_with_edges(mem: Memory):
    p = mem.land("Databases", mark_access=False)
    labels = {d.label for d in p.signpost}
    assert "B-trees" in labels and "Vacuum tuning" in labels
    verbs = {d.verb for d in p.signpost}
    assert "depends_on" in verbs


def test_steer_rides_edge_and_counts_traversal(mem: Memory):
    before = mem.nodes[mem._resolve("B-trees").id].breadth.traversal_count
    p = mem.steer("Databases", "B-trees")
    assert "[B-trees]" in " ".join(p.path_so_far)
    after = mem.nodes[mem._resolve("B-trees").id].breadth.traversal_count
    assert after == before + 1
    # persisted to disk too
    reloaded = Memory(str(mem.store.root))
    assert (reloaded.nodes[mem._resolve("B-trees").id]
            .breadth.traversal_count == after)


def test_write_triple_idempotent(mem: Memory):
    n_before = len(mem.edges)
    mem.write_triple("Databases", "depends_on", "B-trees")
    assert len(mem.edges) == n_before  # duplicate refused silently


def test_lca_of_siblings(mem: Memory):
    # B-trees and Vacuum tuning live in the same graph to graph is common ground
    ctx = mem.lca_context(["B-trees", "Vacuum tuning"])
    assert ctx["lca_kind"] == "graph"
    assert ctx["lca_name"] == "Test"


def test_lca_through_nested_graph(mem: Memory, ws: str):
    from miea_mem.model import Graph, new_id

    # give Postgres a child graph containing a leaf
    pg = mem._resolve("Postgres")
    child = Graph(id=new_id(), name="PG internals",
                  parent_node_id=pg.id)
    mem.graphs[child.id] = child
    mem.store.save_graph(child)
    pg.child_graph_id = child.id
    mem.parent_of.clear()
    for gid, g in mem.graphs.items():
        for nid in g.node_ids:
            mem.parent_of[nid] = (gid, g.parent_node_id)
    wal = mem.create_node("WAL", under_graph=child.id)

    ctx = mem.lca_context(["Postgres", "WAL"])
    assert ctx["lca_kind"] == "node"
    assert ctx["lca_name"] == "Postgres"


def test_query_scoped_limits_to_subtree(mem: Memory):
    hits = mem.query_scoped("Test", "vacuum")
    assert hits and hits[0].node.label == "Vacuum tuning"


def test_forget_removes_edges_and_persists(mem: Memory, ws: str):
    removed = mem.forget("Databases")
    assert removed == 1
    fresh = Memory(ws)
    assert "Databases" not in {n.label for n in fresh.nodes.values()}
    # its edges are gone as well
    assert not any(e.verb == "depends_on" for e in fresh.edges.values())


def test_breadth_order_only_never_filters(mem: Memory):
    # hammer access on one node; others must remain retrievable
    for _ in range(10):
        mem.land("B-trees", mark_access=True)
    hot = mem.breadth_score(mem._resolve("B-trees").id)
    cold = mem.breadth_score(mem._resolve("Vacuum tuning").id)
    assert hot > cold
    # cold node still landable, rank affects order, never access
    p = mem.land("Vacuum tuning", mark_access=False)
    assert p.node.label == "Vacuum tuning"


def test_refresh_if_changed_picks_up_external_writes(mem: Memory, ws):
    # simulate another process (CLI/second agent) writing to the workspace
    other = Memory(ws)
    other.create_node("WrittenByOther")
    # our long-lived instance hasn't reloaded yet
    assert "WrittenByOther" not in {n.label for n in mem.nodes.values()}
    mem.refresh_if_changed()
    assert "WrittenByOther" in {n.label for n in mem.nodes.values()}


def test_payload_shows_dates(mem: Memory):
    p = mem.land("Databases", mark_access=False)
    assert "first:" in p.render()
