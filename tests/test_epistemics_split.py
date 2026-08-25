"""Tests for epistemics, provenance, promotion-split, paging."""

import pytest

from miea_mem.core import Memory
from miea_mem.epistemics import (
    CONTRADICTED,
    CORROBORATED,
    CONTESTED,
    EpistemicPass,
    UNVERIFIABLE,
    UNVERIFIED,
    NullVerifier,
    classify_claim,
    make_serp_verifier,
)
from miea_mem.store import Store


@pytest.fixture()
def mem(tmp_path) -> Memory:
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    return Memory(str(root))


# -- claim classification ----------------------------------------------------


def test_classify_world_vs_user_vs_opaque():
    assert classify_claim("the earth is flat") == "world"
    assert classify_claim("prefers dark themes for coding") == "opaque"
    assert classify_claim("hi") == "opaque"


# -- epistemic pass -----------------------------------------------------------


def test_pass_corroborates_claim(mem: Memory):
    def fake_serp(q):
        return [{"title": "Science confirms shape of planets"}]

    v = make_serp_verifier(fake_serp)
    claim = mem.create_node("flat earth", content="the earth is flat",
                            type="claim")
    claim.epistemic_status = UNVERIFIED
    mem.store.save_node(claim)

    report = EpistemicPass(mem, v).run()
    assert report and report[0]["status"] == CORROBORATED
    assert mem.nodes[claim.id].epistemic_status == CORROBORATED
    # no source edges for plain corroboration
    assert not any(e.verb in ("contradicted_by", "some_sources_say")
                   for e in mem.edges.values())


def test_pass_contradicts_and_creates_source_nodes(mem: Memory):
    def fake_serp(q):
        return [
            {"title": "The myth that the earth is not round"},
            {"title": "Flat earth debunked: it is a spheroid"},
            {"title": "False belief: earth is flat"},
        ]

    claim = mem.create_node("flat earth", content="the earth is flat",
                            type="claim")
    claim.epistemic_status = UNVERIFIED
    mem.store.save_node(claim)

    report = EpistemicPass(mem, make_serp_verifier(fake_serp)).run()
    assert report[0]["status"] == CONTRADICTED
    assert mem.nodes[claim.id].epistemic_status == CONTRADICTED
    cb = [e for e in mem.edges.values() if e.verb == "contradicted_by"]
    assert len(cb) == 3  # one edge per disputing source
    # signpost now shows the contradiction inline
    p = mem.land("flat earth", mark_access=False)
    rendered = p.render()
    assert "contradicted" in rendered


def test_mixed_evidence_becomes_contested_with_plural_edges(mem: Memory):
    def fake_serp(q):
        return [
            {"title": "Study supports the claim"},
            {"title": "Why this claim is wrong"},
        ]

    claim = mem.create_node("six meals", content="eating six meals boosts metabolism",
                            type="claim")
    claim.epistemic_status = UNVERIFIED
    mem.store.save_node(claim)
    report = EpistemicPass(mem, make_serp_verifier(fake_serp)).run()
    assert report[0]["status"] == CONTESTED
    verbs = {e.verb for e in mem.edges.values()}
    assert "some_sources_say" in verbs


def test_null_verifier_marks_unverifiable_and_user_domain_skipped(mem: Memory):
    pref = mem.create_node("dark themes")
    pref.epistemic_status = UNVERIFIED  # even if mislabeled…
    mem.store.save_node(pref)
    claim = mem.create_node("weird claim", content="quantum woo does things",
                            type="claim")
    claim.epistemic_status = UNVERIFIED
    mem.store.save_node(claim)

    report = EpistemicPass(mem, NullVerifier()).run()
    statuses = {r["status"] for r in report}
    assert statuses == {UNVERIFIABLE}
    # user-domain node ("dark themes") isn't lookupable → stays out of the pass
    assert mem.nodes[pref.id].epistemic_status == UNVERIFIED
    # world claim got annotated by the (null) verifier
    assert mem.nodes[claim.id].epistemic_status == UNVERIFIABLE


def test_system_verbs_reserved_from_user_writes(mem: Memory):
    with pytest.raises(ValueError, match="reserved for the verify pass"):
        mem.write_triple("flat earth", "contradicted_by", "NASA",
                         create_missing=True)


def test_provenance_report_tracks_backing(mem: Memory):
    user = mem.create_node("Kushal", type="anchor")
    claim = mem.create_node("likes rust")
    mem.write_triple(user.label, "user_asserts", claim.label)
    orphan = mem.create_node("unbacked fact")

    rep = mem.provenance_report()
    labels = [u["label"] for u in rep["unbacked"]]
    assert "likes rust" not in labels   # has user_asserts backing
    assert "unbacked fact" in labels
    assert rep["backed"] >= 1


# -- promotion-split ----------------------------------------------------------


def _make_fanout(mem: Memory, hub: str, n: int):
    """n children under hub's graph; half linked with verb 'uses', half free."""
    mem.create_node(hub)
    for i in range(n):
        label = f"child{i}"
        mem.create_node(label)
        if i % 2 == 0:
            mem.write_triple(hub, "uses", label)


def test_split_refuses_when_under_cap(mem: Memory):
    _make_fanout(mem, "hub0", 4)
    created = mem.split_if_overloaded(mem._resolve("hub0").id, cap=9)
    assert created == []


def test_split_promotes_groups_by_verb_signature(mem: Memory):
    _make_fanout(mem, "hub1", 12)  # 6 'uses'-linked + 6 unlinked
    hub_id = mem._resolve("hub1").id
    before = len(mem.nodes)
    created = mem.split_if_overloaded(hub_id, cap=5)
    assert len(created) >= 2          # two groups promoted
    after = mem.fanout(hub_id)
    assert after <= 5                 # pressure relieved
    # group anchors exist and own sub-graphs containing the members
    for gid in created:
        grp = mem.nodes[gid]
        assert grp.type == "anchor"
        assert grp.child_graph_id in mem.graphs
    # members still reachable via signpost of the group nodes
    grp = mem.nodes[created[0]]
    payload = mem.land(grp.label, mark_access=False)
    assert payload.node.label == grp.label
    assert len(mem.nodes) == before + len(created)


def test_split_persists(mem: Memory, tmp_path):
    _make_fanout(mem, "hub2", 12)
    created = mem.split_if_overloaded(mem._resolve("hub2").id, cap=5)
    fresh = Memory(str(mem.store.root))
    for gid in created:
        assert gid in fresh.nodes
        assert fresh.nodes[gid].child_graph_id in fresh.graphs


# -- signpost epistemic status -------------------------------------------------


def test_destination_shows_epistemic_status(mem: Memory):
    claim = mem.create_node("dubious", content="some dubious world thing is so",
                            type="claim")
    claim.epistemic_status = CONTRADICTED
    mem.store.save_node(claim)
    other = mem.create_node("plain")
    mem.write_triple(other.label, "relates_to", claim.label)
    p = mem.land(other.label, mark_access=False)
    line = [d.render() for d in p.signpost if d.label == "dubious"][0]
    assert "contradicted" in line


# -- signpost paging ----------------------------------------------------------


def test_signpost_paging(mem: Memory):
    hub = mem.create_node("pager")
    for i in range(12):
        mem.write_triple("pager", "relates_to", f"p{i}", create_missing=True)
    p0 = mem.land("pager", mark_access=False)
    assert len(p0.signpost) == 7          # TOP_K page 0
    assert p0.total_destinations == 12
    assert p0.page == 0
    p1 = mem.land("pager", mark_access=False, page=1)
    assert 0 < len(p1.signpost) <= 7      # remainder page
    assert p1.page == 1
    labels = {d.label for d in p0.signpost}
    assert labels.isdisjoint({d.label for d in p1.signpost})


# -- write placement ------------------------------------------------------------


def test_placement_hint_same_domain_vs_cross_branch(mem: Memory, tmp_path):
    mem.create_node("Postgres")
    mem.create_node("WAL")
    # same-domain pair → leaf level
    hint = mem.placement_hint("Postgres", "WAL")
    assert "leaf" in hint["reason"]


def test_placement_hint_cross_branch_points_at_ancestor(mem: Memory):
    mem.create_node("Postgres")
    # build nested structure: Postgres owns child graph with WAL inside
    from miea_mem.model import Graph, new_id
    pg = mem._resolve("Postgres")
    child = Graph(id=new_id(), name="PG internals", parent_node_id=pg.id)
    mem.graphs[child.id] = child
    pg.child_graph_id = child.id
    wal = mem.create_node("WAL2", under_graph=child.id)

    # WAL2 (under Postgres) vs MySQL — wait: use a node INSIDE Postgres's own
    # sibling branch so Postgres is genuinely the shared ancestor.
    mysql = mem.create_node("MySQL")
    mem.write_triple(mysql.label, "competes_with", "Postgres")
    hint = mem.placement_hint(wal.label, "Postgres")
    # WAL2 is contained BY Postgres → direct ancestry, LCA is the node itself
    assert hint["suggest"].startswith("under [Postgres]")
