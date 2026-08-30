# Core engine, Handles the search.Loads the workspace files into flat tables, builds derived
# indexes, and implements every operation: search, landing, steering,
# scoped queries, LCA, writes, deletion, splitting.

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .model import Breadth, DivergenceEntry, Edge, Graph, Manifest, Node, new_id
from .store import Store

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its of on or "
    "that the this to was were will with".split()
)

TOP_K = 7  # destinations shown per signpost page


def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9_]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def _parse_iso(s: str | None, cache: dict[str, float] | None = None) -> float | None:
    # Parsing dates is hot inside signpost scoring. When a cache dict is
    # passed it is used and filled; callers holding many nodes pass one.
    if not s:
        return None
    if cache is not None:
        hit = cache.get(s)
        if hit is not None:
            return hit
    try:
        ts = datetime.fromisoformat(s).timestamp()
        if cache is not None:
            cache[s] = ts
        return ts
    except ValueError:
        return None


def _fmt_date(s: str | None) -> str:
    # Human readable date. Agents use this for recency reasoning.
    ts = _parse_iso(s)
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


# Payload shapes

@dataclass
class Destination:
    # One signpost entry: a reference to a reachable node, never a summary.
    node_id: str
    label: str
    verb: str | None          # edge verb when reached via an edge
    direction: str | None     # "out", "in" or "child"
    score: float = 0.0
    epistemic_status: str | None = None
    cue_node_id: str | None = None   # map anchor rows: hottest leaf below
    cue_label: str | None = None     # the branch, shown as a routing cue

    def render(self) -> str:
        if self.direction == "out":
            via = f" --{self.verb}--> "
        elif self.direction == "in":
            via = f" <--{self.verb}-- "
        else:
            via = " > "
        line = f"[{self.label}]{via}"
        if self.cue_label:
            line += f"cue: [{self.cue_label}] "
        line += f"(score {self.score:.2f}"
        if self.epistemic_status and self.epistemic_status != "unverifiable":
            line += f", {self.epistemic_status}"
        return line + ")"


@dataclass
class Payload:
    # What an agent receives on landing: current content plus signpost.

    node: Node
    path_so_far: list[str] = field(default_factory=list)
    signpost: list[Destination] = field(default_factory=list)
    total_destinations: int = 0   # full count, may exceed what is shown
    page: int = 0                 # which TOP_K slice of the signpost
    matched_edges: list[dict] = field(default_factory=list)
    transit_notes: list[dict] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"* [{self.node.label}] ({self.node.type})"]
        if self.node.content:
            lines.append(f"  {self.node.content}")
        if self.path_so_far:
            lines.append(f"  sentence-so-far: {' '.join(self.path_so_far)}")
        if self.node.child_graph_id:
            lines.append("  (has nested graph)")
        created = _fmt_date(self.node.created_at)
        updated = _fmt_date(self.node.updated_at)
        stamp = f"  first: {created}"
        if updated and updated != created:
            stamp += f", last updated {updated}"
        lines.append(stamp)
        lines.append(f"  epistemic: {self.node.epistemic_status}")
        if self.signpost:
            lines.append("  destinations:")
            for d in self.signpost:
                lines.append(f"    {d.render()}")
            hidden = self.total_destinations - len(self.signpost)
            if hidden > 0:
                lines.append(
                    f"    {hidden} more exist. Use a scoped query or page "
                    "further to reach them.")
        if self.transit_notes:
            lines.append("  slid past (one land(id) away if one fits):")
            for note in self.transit_notes:
                lines.append(f"    [{note['label']}] {note['snippet']}")
        return "\n".join(lines)


# Memory

class Memory:
    def __init__(self, root: str, *, embedder: Any | None = "auto"):
        self.store = Store(root)
        self.manifest: Manifest = self.store.load_manifest()

        # Flat tables. Every entity sits side by side keyed by id.
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.graphs: dict[str, Graph] = {}

        # Derived indexes, rebuildable from the flat tables at any time.
        self.out_edges: dict[str, list[str]] = {}
        self.in_edges: dict[str, list[str]] = {}
        self.parent_of: dict[str, tuple[str, str | None]] = {}
        self._fts: dict[str, Counter] = {}
        self._dirty_maps: set[str] = set()
        self._ts_cache: dict[str, float] = {}  # iso string to epoch seconds

        # Semantic layer. Optional. Absent embedder means keyword search only.
        self._vector_index: Any | None = None
        if embedder == "auto":
            from .semantic import try_load_embedder

            emb = try_load_embedder()
            if emb is not None:
                from .semantic import VectorIndex

                self._vector_index = VectorIndex(self.store.root, emb)
        elif embedder is not None:
            from .semantic import VectorIndex

            self._vector_index = VectorIndex(self.store.root, embedder)

        self._load()

        if self._vector_index is not None:
            self._vector_index.ensure_all(self.nodes)
        self._fingerprint = self.store.fingerprint()

    def _note_self_write(self) -> None:
        # Our own writes change file mtimes. Without this, the next read
        # would mistake them for an external writer and reload everything.
        self._fingerprint = self.store.fingerprint()

    # Loading

    def _load(self) -> None:
        for g in self.store.all_graphs():
            self.graphs[g.id] = g
        for n in self.store.all_nodes():
            self.nodes[n.id] = n
            self._index_node(n)
        for e in self.store.all_edges():
            self.edges[e.id] = e
            self.out_edges.setdefault(e.source_id, []).append(e.id)
            self.in_edges.setdefault(e.target_id, []).append(e.id)
        # Containment overlay built last because it needs graphs and nodes.
        for gid, g in self.graphs.items():
            for nid in g.node_ids:
                self.parent_of[nid] = (gid, g.parent_node_id)

    def _index_node(self, n: Node) -> None:
        text = " ".join([n.label, n.type, " ".join(n.tags), n.content])
        self._fts[n.id] = Counter(_tokenize(text))

    def reindex(self) -> None:
        # Wipe every derived index and rebuild from the flat tables.
        self._fts.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self.parent_of.clear()
        for n in self.nodes.values():
            self._index_node(n)
        if self._vector_index is not None:
            self._vector_index.vectors.clear()
            self._vector_index.ensure_all(self.nodes)
            self._vector_index.save()
        for e in self.edges.values():
            self.out_edges.setdefault(e.source_id, []).append(e.id)
            self.in_edges.setdefault(e.target_id, []).append(e.id)
        for gid, g in self.graphs.items():
            for nid in g.node_ids:
                self.parent_of[nid] = (gid, g.parent_node_id)
        # every node re-evaluates its map lazily on next access
        self._dirty_maps = set(self.nodes)

    def refresh_if_changed(self) -> None:
        # Reloads when another process modified the workspace. An MCP server
        # lives long while CLI writers come and go; a stat fingerprint per
        # call detects their writes cheaply.
        fp = self.store.fingerprint()
        if fp != self._fingerprint:
            self.nodes.clear()
            self.edges.clear()
            self.graphs.clear()
            self.out_edges.clear()
            self.in_edges.clear()
            self.parent_of.clear()
            self._fts.clear()
            self._dirty_maps.clear()
            self._load()
            self._fingerprint = fp

    # Ranking

    def breadth_score(self, node_id: str, now: float | None = None) -> float:
        # Access count plus traversal count, log compressed, plus a recency
        # boost of up to two points over fourteen days. Orders results.
        # Never used as a filter.
        n = self.nodes[node_id]
        now = now or datetime.now(timezone.utc).timestamp()
        access = n.breadth.access_count + n.breadth.traversal_count
        recency_boost = 0.0
        ts = _parse_iso(n.breadth.last_accessed, self._ts_cache) or \
            _parse_iso(n.created_at, self._ts_cache)
        if ts:
            age_days = max(0.0, (now - ts) / 86400)
            recency_boost = math.exp(-age_days / 14.0) * 2.0
        return math.log1p(access) + recency_boost

    # Resolution helpers

    def _resolve(self, ref: str, *, create: bool = False) -> Node:
        # Id first, then unique label. With create=True unknown refs become
        # new nodes; without it unknown refs raise.
        if ref in self.nodes:
            return self.nodes[ref]
        matches = [n for n in self.nodes.values() if n.label.lower() == ref.lower()]
        if len(matches) == 1:
            return matches[0]
        if create:
            return self.create_node(label=ref)
        raise LookupError(f"no node matching {ref!r}")

    def _resolve_graph(self, ref: str) -> Graph:
        if ref in self.graphs:
            return self.graphs[ref]
        matches = [g for g in self.graphs.values() if g.name.lower() == ref.lower()]
        if len(matches) == 1:
            return matches[0]
        available = ", ".join(
            f"'{g.name}'" for g in self.graphs.values()) or "(none)"
        raise LookupError(
            f"no graph matching {ref!r}. Available graphs: {available}. "
            "under_graph is optional, leave it out to file at root.")

    # Signpost construction

    def _all_destinations(self, node: Node) -> list[Destination]:
        import time as _time

        now = _time.time()
        dests: list[Destination] = []
        # the signpost source self-heals: dirty forks rebuild before any
        # row is read, so direct callers see fresh maps too
        self._refresh_divergence_map(node)

        # Neighbors reached by edges, both directions.
        for eid in self.out_edges.get(node.id, []):
            e = self.edges[eid]
            other = self.nodes.get(e.target_id)
            if other:
                dests.append(Destination(other.id, other.label, e.verb, "out",
                                         self.breadth_score(other.id, now),
                                         other.epistemic_status))
        for eid in self.in_edges.get(node.id, []):
            e = self.edges[eid]
            other = self.nodes.get(e.source_id)
            if other:
                dests.append(Destination(other.id, other.label, e.verb, "in",
                                         self.breadth_score(other.id, now),
                                         other.epistemic_status))

        # Branch tier: the stored divergence map when the fork has one,
        # live graph membership otherwise. One destination per member;
        # anchor rows carry their cue leaf and score by the hotter of
        # the two, so fresh leaves lift their branch immediately.
        g = self._tier_graph(node)
        if node.divergence_map:
            for entry in node.divergence_map:
                m = self.nodes.get(entry.node_id)
                if not m:
                    continue
                cue_id = entry.cue_leaf_id
                if cue_id and cue_id not in self.nodes:
                    cue_id = None
                score = self.breadth_score(m.id, now)
                if cue_id:
                    score = max(score, self.breadth_score(cue_id, now))
                dests.append(Destination(
                    m.id, m.label, None, "child", score,
                    m.epistemic_status,
                    cue_node_id=cue_id,
                    cue_label=entry.cue_label if cue_id else None))
        elif g:
            for nid in g.node_ids:
                if nid == node.id or nid not in self.nodes:
                    continue
                dests.append(Destination(nid, self.nodes[nid].label, None,
                                         "child", self.breadth_score(nid, now),
                                         self.nodes[nid].epistemic_status))

        # One destination per node. Edge view wins over sibling view.
        seen: set[str] = set()
        deduped: list[Destination] = []
        for d in dests:
            if d.node_id not in seen:
                seen.add(d.node_id)
                deduped.append(d)
        deduped.sort(key=lambda d: d.score, reverse=True)
        return deduped

    def _payload(self, node: Node, *, include_edges: bool = False,
                 page: int = 0) -> Payload:
        all_dests = self._all_destinations(node)
        total = len(all_dests)
        start = page * TOP_K
        payload = Payload(node=node,
                          signpost=all_dests[start:start + TOP_K],
                          total_destinations=total)
        payload.page = page
        if include_edges:
            payload.matched_edges = [
                {"id": eid, "verb": self.edges[eid].verb}
                for eid in self.out_edges.get(node.id, [])
            ]
        return payload

    def _find_destination(self, node: Node, want: str) -> Destination | None:
        # Search the visible page first, then the full set. Steering can
        # reach any real destination even beyond top-k.
        for d in self._all_destinations(node):
            if d.node_id == want or d.label.lower() == want.lower():
                return d
        return None

    # Scope helpers

    def _subtree_nodes(self, graph_id: str) -> set[str]:
        seen = set()
        stack = [graph_id]
        while stack:
            g = self.graphs.get(stack.pop())
            if not g:
                continue
            for nid in g.node_ids:
                if nid not in seen:
                    seen.add(nid)
                    n = self.nodes.get(nid)
                    if n and n.child_graph_id:
                        stack.append(n.child_graph_id)
        return seen

    def _register_in_graph(self, entity_id: str, near_node_id: str,
                           *, is_edge: bool) -> None:
        entry = self.parent_of.get(near_node_id)
        gid = entry[0] if entry else self.manifest.root_graph_id
        g = self.graphs[gid]
        if is_edge:
            g.edge_ids.add(entity_id)
        else:
            g.node_ids.add(entity_id)
        self.store.save_graph(g)

    def _graph_containing_edge(self, e: Edge) -> Graph | None:
        for g in self.graphs.values():
            if e.id in g.edge_ids:
                return g
        return None

    def _mark_dirty(self, node_id: str) -> None:
        # Flags every fork whose divergence map a structural change at
        # node_id invalidates: ancestors via parent pointers, plus for
        # each graph we own or live in its owning node and tier-mates
        # whose view spans that graph. Bounded by the fanout cap.
        entry = self.parent_of.get(node_id)
        while entry:
            gid, pid = entry
            if pid:
                self._dirty_maps.add(pid)
                entry = self.parent_of.get(pid)
            else:
                break
        node = self.nodes.get(node_id)
        gids = []
        if node and node.child_graph_id:
            gids.append(node.child_graph_id)   # the graph we own
        lived_in = self.parent_of.get(node_id, (None,))[0]
        if lived_in:
            gids.append(lived_in)              # the graph we live in
        for gid in gids:
            g = self.graphs.get(gid)
            if not g:
                continue
            if g.parent_node_id:
                self._dirty_maps.add(g.parent_node_id)
            for nid in g.node_ids:
                n = self.nodes.get(nid)
                if n and not n.child_graph_id:
                    self._dirty_maps.add(nid)

    # Divergence maps

    def _tier_graph(self, node: Node) -> Graph | None:
        # The branch tier a node is viewed over: the graph it owns, else
        # the graph it lives in.
        gid = node.child_graph_id or (
            self.parent_of.get(node.id, (None,))[0])
        return self.graphs.get(gid) if gid else None

    def _map_warranted(self, node: Node) -> bool:
        # A stored map only adds information when the tier contains
        # anchors; otherwise its entries equal the live sibling view and
        # the signpost computes them anyway.
        g = self._tier_graph(node)
        if not g:
            return False
        return any(nid != node.id and nid in self.nodes
                   and self.nodes[nid].child_graph_id
                   for nid in g.node_ids)

    def _build_divergence_map(self, node: Node) -> list[DivergenceEntry]:
        # One entry per tier member: anchors route with a cue to their
        # hottest leaf, singletons route to themselves. Stored order is
        # deterministic; ranking happens live from breadth at read time.
        g = self._tier_graph(node)
        if not g:
            return []
        now = datetime.now(timezone.utc).timestamp()
        entries: list[DivergenceEntry] = []
        for nid in g.node_ids:
            m = self.nodes.get(nid)
            if not m or nid == node.id:
                continue
            if m.child_graph_id:
                candidates = sorted(self._subtree_nodes(m.child_graph_id))
                cues = sorted(
                    (cid for cid in candidates if cid in self.nodes),
                    key=lambda cid: (-self.breadth_score(cid, now), cid))
                cue = self.nodes[cues[0]] if cues else None
                entries.append(DivergenceEntry(
                    node_id=m.id, label=m.label, kind="anchor",
                    cue_leaf_id=cue.id if cue else None,
                    cue_label=cue.label if cue else None))
            else:
                entries.append(DivergenceEntry(
                    node_id=m.id, label=m.label, kind="leaf"))
        entries.sort(key=lambda e: (e.kind, e.label))
        return entries

    def _refresh_divergence_map(self, node: Node) -> None:
        # Lazy regeneration per the write policy: dirty forks rebuild on
        # their next access; cold branches keep cheap stale maps. A fork
        # that never had a map builds one on first view - an empty map
        # on a warranted fork means "not yet computed", not "stale".
        if node.id in self._dirty_maps:
            self._dirty_maps.discard(node.id)
            self._rebuild_map(node)
        elif not node.divergence_map and self._map_warranted(node):
            self._rebuild_map(node)

    def _rebuild_map(self, node: Node) -> None:
        if not self._map_warranted(node):
            if node.divergence_map:
                node.divergence_map = []
                self.store.save_node(node)
                self._note_self_write()
            return
        entries = self._build_divergence_map(node)
        if node.divergence_map != entries:
            node.divergence_map = entries
            self.store.save_node(node)
            self._note_self_write()

    # Search

    def search(self, query: str, limit: int = 5) -> list[tuple[Node, float]]:
        # Hybrid entry point search. Keyword leg catches exact terms,
        # vector leg catches paraphrases, reciprocal rank fusion merges
        # them. Without an embedder only keywords run. Order only.
        self.refresh_if_changed()
        q = Counter(_tokenize(query))

        fts_ids: list[str] = []
        if q:
            scored: list[tuple[float, str]] = []
            for nid, tokens in self._fts.items():
                overlap = sum((tokens & q).values())
                if overlap:
                    norm = sum(tokens.values()) or 1
                    scored.append((overlap / math.sqrt(norm), nid))
            scored.sort(reverse=True)
            fts_ids = [nid for _, nid in scored]

        vec_ids: list[str] = []
        if self._vector_index is not None:
            try:
                self._vector_index.ensure_all(self.nodes)
                hits = self._vector_index.query(query, k=limit * 3)
                # Cosine floor keeps weak matches out of the fusion.
                vec_ids = [nid for nid, s in hits if s >= 0.35]
            except Exception:
                vec_ids = []  # degrade to keywords, never crash

        if not fts_ids and not vec_ids:
            return []

        from .semantic import rrf_fuse

        lists = [l for l in (fts_ids, vec_ids) if l]
        fused = rrf_fuse(lists, top=max(limit * 2, len(fts_ids)))
        out = []
        for nid, _ in fused[:limit]:
            out.append((self.nodes[nid], self.breadth_score(nid)))
        return out

    # Read loop

    def land(self, ref: str, *, include_edges: bool = False,
             mark_access: bool = True, page: int = 0) -> Payload:
        # Arrive at a node. Returns its content plus its signpost.
        # mark_access feeds the ranking signals; tests pass False.
        self.refresh_if_changed()
        node = self._resolve(ref)
        if mark_access:
            from .model import now_iso

            node.breadth.access_count += 1
            node.breadth.last_accessed = now_iso()
            self.store.save_node(node)
            self._update_cue(node.id)
            self._note_self_write()
        return self._payload(node, include_edges=include_edges, page=page)

    def _update_cue(self, node_id: str) -> None:
        # The recency tiebreaker, live: a leaf that outraces its branch's
        # current cue becomes the cue, so fresh leaves surface on the
        # signpost immediately - without waiting for a structural write
        # that would rebuild the whole map. O(fanout): discovers the fork
        # (nested: the anchor's parent node; root fork: a tier-mate in
        # the shared graph whose map names the anchor), then swaps.
        entry = self.parent_of.get(node_id)
        if not entry:
            return
        anchor_id = entry[1]
        anchor = self.nodes.get(anchor_id) if anchor_id else None
        if not anchor or anchor.type != "anchor":
            return                      # the parent is not a route anchor
        fork = self._fork_of_anchor(anchor_id)
        anchor_entry = (next((e for e in fork.divergence_map
                              if e.node_id == anchor_id), None)
                        if fork else None)
        if anchor_entry is None or anchor_entry.cue_leaf_id is None:
            return
        if anchor_entry.cue_leaf_id == node_id:
            return                      # already the cue
        if self.breadth_score(node_id) <= self.breadth_score(
                anchor_entry.cue_leaf_id):
            return                      # stored cue still outranks it
        anchor_entry.cue_leaf_id = node_id
        anchor_entry.cue_label = self.nodes[node_id].label
        self.store.save_node(fork)
        self._note_self_write()

    def _fork_of_anchor(self, anchor_id: str) -> Node | None:
        # The node whose divergence map names this anchor. Nested forks:
        # the anchor's parent node. Root forks: a tier-mate in the graph
        # the anchor lives in (its own parent chain stops at the root
        # graph), scanned by membership - bounded by the fanout cap.
        parent_id = self.parent_of.get(anchor_id, (None, None))[1]
        parent = self.nodes.get(parent_id) if parent_id else None
        if parent is not None:
            return parent
        gid = self.parent_of.get(anchor_id, (None, None))[0]
        g = self.graphs.get(gid) if gid else None
        if not g:
            return None
        for nid in g.node_ids:
            n = self.nodes.get(nid)
            if n and any(e.node_id == anchor_id
                         for e in n.divergence_map):
                return n
        return None

    def steer(self, ref: str, destination: str, *,
              include_edges: bool = False) -> Payload:
        # Move across one named edge to a signpost destination. Traversal
        # counts rise on both ends and the path accumulates as a sentence.
        self.refresh_if_changed()
        node = self._resolve(ref)
        dest = self._find_destination(node, destination)
        if dest is None:
            raise LookupError(f"no destination matching {destination!r}")

        nxt = self.nodes[dest.node_id]
        from .model import now_iso

        for n in (node, nxt):
            n.breadth.traversal_count += 1
            n.breadth.last_accessed = now_iso()
            self.store.save_node(n)
            self._update_cue(n.id)
        self._note_self_write()

        payload = self._payload(nxt, include_edges=include_edges)
        payload.path_so_far = [f"[{node.label}]"]
        if dest.verb:
            payload.path_so_far.append(f"--{dest.verb}-->")
        payload.path_so_far.append(f"[{nxt.label}]")
        return payload

    def slide(self, ref: str, destination: str, *, deep: bool = False,
              query: str | None = None,
              mark_access: bool = True) -> Payload:
        # The waterslide: ride a chosen branch in one pass instead of
        # steering level by level. Without deep, lands at the branch
        # entry (no skips). With deep, rides on to the entry's cue leaf,
        # recording every skipped node as a transit note so the
        # arrival-time sufficiency check can glance backward. The chain
        # concatenates into a noun-verb-noun proposition; the judgment
        # of "enough?" stays with the consuming agent.
        self.refresh_if_changed()
        node = self._resolve(ref)
        entry = self._find_destination(node, destination)
        if entry is None:
            raise LookupError(f"no destination matching {destination!r}")
        target = entry.node_id
        if deep:
            m_entry = next((e for e in node.divergence_map
                            if e.node_id == target), None)
            if m_entry and m_entry.cue_leaf_id in self.nodes:
                target = m_entry.cue_leaf_id
        # ride from ref to the target: straight down the chain ref owns,
        # or one lateral hop when the target sits in ref's own tier (the
        # root fork's branches are its tier-mates, not descendants)
        chain = [target]
        cur = target
        home = node.child_graph_id or (
            self.parent_of.get(node.id, (None,))[0])
        while cur != node.id:
            gid, pid = self.parent_of.get(cur, (None, None))
            if home and gid == home:
                break                     # entered ref's tier: hop on
            if not pid:
                raise LookupError(
                    f"{destination!r} is not reachable beneath or beside "
                    f"{ref!r}; use steer for edge hops")
            cur = pid
            chain.append(cur)
        if cur != node.id:
            chain.append(node.id)         # the lateral hop onto the branch
        chain.reverse()                   # [ref, ..., landing]

        from .model import now_iso

        now = now_iso()
        for nid in chain:
            n = self.nodes[nid]
            n.breadth.traversal_count += 1
            n.breadth.last_accessed = now
            self.store.save_node(n)
        landing = self.nodes[chain[-1]]
        if mark_access:
            landing.breadth.access_count += 1
            landing.breadth.last_accessed = now
            self.store.save_node(landing)
        self._update_cue(landing.id)
        self._note_self_write()

        payload = self._payload(landing)
        q = Counter(_tokenize(query)) if query else None
        path: list[str] = []
        notes: list[dict] = []
        prev: str | None = None
        for i, nid in enumerate(chain):
            n = self.nodes[nid]
            if prev is not None:
                verb = None
                for eid in self.out_edges.get(prev, []):
                    if self.edges[eid].target_id == nid:
                        verb = self.edges[eid].verb
                        break
                if verb is None:
                    for eid in self.in_edges.get(nid, []):
                        if self.edges[eid].source_id == prev:
                            verb = self.edges[eid].verb
                            break
                path.append(f"--{verb}-->" if verb else ">")
            path.append(f"[{n.label}]")
            if 0 < i < len(chain) - 1:
                overlap = 0.0
                if q:
                    overlap = float(sum(
                        (self._fts.get(nid, Counter()) & q).values()))
                notes.append({"node_id": nid, "label": n.label,
                              "snippet": (n.content or "")[:120],
                              "overlap": overlap})
            prev = nid
        if q:
            notes.sort(key=lambda note: (-note["overlap"],
                                         chain.index(note["node_id"])))
        payload.path_so_far = path
        payload.transit_notes = notes[:8]
        return payload

    def route(self, ref: str, query: str, limit: int = 5) -> dict:
        # Direction picking by hybrid match over a fork's branch entries.
        # Candidates are the standing node itself (it may already be the
        # answer) plus its tier routes; each anchor route inherits the
        # best score of itself and its cue leaf, so leaf-specific
        # queries attribute to the branch they live under. Keyword leg,
        # a subtree-scoped vector leg (one query embedding, cosine
        # against route and cue vectors only - never the whole
        # workspace), and a breadth prior fuse through RRF. Rank
        # orders, never filters.
        from .semantic import cosine, rrf_fuse

        self.refresh_if_changed()
        node = self._resolve(ref)
        self._refresh_divergence_map(node)
        now = datetime.now(timezone.utc).timestamp()

        rows: dict[str, dict] = {}

        def add(nid: str, kind: str, cue_label: str | None,
                cue_id: str | None) -> None:
            n = self.nodes.get(nid)
            if n and nid not in rows:
                rows[nid] = {"node_id": nid, "label": n.label, "kind": kind,
                             "cue_label": cue_label, "cue_id": cue_id,
                             "epistemic_status": n.epistemic_status,
                             "breadth": round(self.breadth_score(nid, now),
                                              3)}

        add(node.id, "self", None, None)
        if node.divergence_map:
            for e in node.divergence_map:
                add(e.node_id, e.kind, e.cue_label, e.cue_leaf_id)
        else:
            g = self._tier_graph(node)
            if g:
                for nid in g.node_ids:
                    m = self.nodes.get(nid)
                    if m and nid != node.id:
                        add(nid, "anchor" if m.child_graph_id else "leaf",
                            None, None)

        def cue_ids(nid: str) -> list[str]:
            extra = rows[nid]["cue_id"]
            return [nid] + ([extra] if extra and extra in self.nodes
                            else [])

        ids = list(rows)
        q = Counter(_tokenize(query))

        # keyword leg: best token overlap across the route and its cue
        kw: list[str] = []
        if q:
            scored = []
            for nid in ids:
                best = 0.0
                for cid in cue_ids(nid):
                    tokens = self._fts.get(cid)
                    if tokens:
                        norm = sum(tokens.values()) or 1
                        best = max(
                            best, sum((tokens & q).values()) / math.sqrt(norm))
                if best > 0:
                    scored.append((best, nid))
            scored.sort(reverse=True)
            kw = [nid for _, nid in scored]

        # vector leg: cosine against this tier's vectors only
        vec: list[str] = []
        if self._vector_index is not None:
            try:
                qv = self._vector_index.embedder.embed([query])[0]
                scored = []
                for nid in ids:
                    best = -1.0
                    for cid in cue_ids(nid):
                        v = self._vector_index.vectors.get(cid)
                        if v:
                            best = max(best, cosine(qv, v))
                    if best >= 0.35:
                        scored.append((best, nid))
                scored.sort(reverse=True)
                vec = [nid for _, nid in scored]
            except Exception:
                vec = []               # degrade to keyword, never crash

        # relevance legs fuse through RRF; breadth orders whatever they
        # do not separate, so hot routes win ties without ever outvoting
        # a relevance hit. Unmatched candidates trail, breadth-ordered -
        # rank orders, never filters.
        legs = [leg for leg in (kw, vec) if leg]
        fused: dict[str, float] = {}
        if legs:
            for nid, score in rrf_fuse(legs, top=len(ids)):
                fused[nid] = score
        ordered = sorted(
            ids,
            key=lambda nid: (-fused.get(nid, 0.0),
                             -self.breadth_score(nid, now), nid))

        matched = bool(legs)
        ambiguous = False
        # ambiguity means conflicting relevance evidence: two routes
        # with real votes finishing neck and neck
        if matched and len(ordered) >= 2:
            s1 = fused.get(ordered[0], 0.0)
            s2 = fused.get(ordered[1], 0.0)
            if s1 > 0 and s2 > 0:
                ambiguous = (s1 - s2) / s1 < 0.15

        routes = []
        for nid in ordered[:max(1, min(limit, len(ids)))]:
            r = dict(rows[nid])
            r["score"] = round(fused.get(nid, 0.0), 4)
            routes.append(r)
        return {"node": {"id": node.id, "label": node.label},
                "query": query, "matched": matched,
                "ambiguous": ambiguous, "routes": routes}

    def suggest_split(self, ref: str, query: str, limit: int = 5) -> dict:
        # Read-time answer to the design doc's open question on split
        # triggers and clustering. The fork's own routing signal says
        # when a boundary is unclear (route ties under a real query),
        # and the semantic grouping preview says where the cut would
        # go. Nothing here mutates; running the split stays an explicit
        # write-tier act.
        self.refresh_if_changed()
        node = self._resolve(ref)
        out = self.route(ref, query, limit=limit)
        tied = []
        if out["matched"] and out["routes"]:
            floor = out["routes"][0]["score"] * 0.85
            tied = [r["label"] for r in out["routes"]
                    if r["score"] >= floor]
        preview = None
        if self._vector_index is not None:
            g = self._tier_graph(node)
            if g:
                members = [self.nodes[nid] for nid in g.node_ids
                           if nid != node.id and nid in self.nodes]
                preview = [
                    {"hint": hint, "labels": sorted(m.label for m in ms)}
                    for hint, ms in self._group_members(
                        members, "semantic", 0.5)]
        return {"node": {"id": node.id, "label": node.label},
                "query": query, "matched": out["matched"],
                "ambiguous": out["ambiguous"], "tied_routes": tied,
                "semantic_groups": preview}

    def query_scoped(self, graph_ref: str, query: str,
                     limit: int = 5) -> list[Payload]:
        # Mediated deep dive inside one subtree. Returns answers.
        self.refresh_if_changed()
        g = self._resolve_graph(graph_ref)
        allowed = self._subtree_nodes(g.id)
        q = Counter(_tokenize(query))
        scored = []
        for nid in allowed:
            tokens = self._fts.get(nid)
            if not tokens:
                continue
            overlap = sum((tokens & q).values())
            if overlap:
                norm = sum(tokens.values()) or 1
                scored.append((overlap / math.sqrt(norm), nid))
        scored.sort(reverse=True)
        out = []
        for _, nid in scored[:limit]:
            n = self.nodes[nid]
            n.breadth.access_count += 1
            self.store.save_node(n)
            self._update_cue(n.id)
            out.append(self._payload(n))
        self._note_self_write()
        return out

    def lca_context(self, refs: list[str]) -> dict:
        # Lowest common ancestor by walking parent pointers toward the root.
        # Pure structure: no model, no embeddings.
        resolved = [self._resolve(r) for r in refs]

        def chain(node_id: str) -> list[str]:
            steps = []
            cur = node_id
            while cur in self.parent_of:
                gid, pid = self.parent_of[cur]
                steps.append(f"graph:{gid}")
                if pid:
                    steps.append(pid)
                    cur = pid
                else:
                    break
            return steps

        chains = [chain(n.id) for n in resolved]
        common = None

        # Direct ancestry: if node A appears in B's chain, A is the answer.
        for i, n in enumerate(resolved):
            if any(n.id in chains[j] for j in range(len(chains)) if j != i):
                common = n.id
                break

        if common is None:
            # Chains run leaf to root. Walk both from the root end while
            # they agree; the last agreement is the deepest common ground.
            pairs = zip(reversed(chains[0]),
                        reversed(chains[1]) if len(chains) > 1 else [])
            for a, b in pairs:
                if a == b:
                    common = a
                else:
                    break

        def render(x: str | None) -> str | None:
            if x and x.startswith("graph:"):
                return self.graphs[x[6:]].name
            return self.nodes[x].label if x else None

        return {
            "lca": common,
            "lca_kind": ("graph" if (common or "").startswith("graph:")
                         else "node" if common else None),
            "lca_name": render(common),
            "nodes": [{"id": n.id, "label": n.label} for n in resolved],
        }

    # Write tier

    def write_triple(self, source_ref: str, verb: str, target_ref: str,
                     *, create_missing: bool = False,
                     provenance: str | None = None) -> Edge:
        # Adds a named edge between two nodes. Duplicate triples are
        # idempotent. Reserved verbs are rejected so user writes can never
        # fake corroboration. Optional provenance records who asserts it.
        from .epistemics import PROVENANCE_VERBS, SYSTEM_VERBS

        if verb in SYSTEM_VERBS:
            raise ValueError(
                f"verb {verb!r} is reserved for the verify pass")
        if not re.fullmatch(r"[a-z][a-z0-9_]*(_[a-z0-9_]+)*", verb):
            raise ValueError(
                f"verb {verb!r} must be a lowercase_snake phrase, "
                "example: persists_with")
        if provenance is not None and provenance not in PROVENANCE_VERBS:
            raise ValueError(
                f"provenance must be one of {sorted(PROVENANCE_VERBS)}")

        s = self._resolve(source_ref, create=create_missing)
        t = self._resolve(target_ref, create=create_missing)

        duplicate = any(
            self.edges[eid].target_id == t.id and self.edges[eid].verb == verb
            for eid in self.out_edges.get(s.id, []))
        if duplicate and not provenance:
            return next(
                self.edges[eid] for eid in self.out_edges.get(s.id, [])
                if self.edges[eid].target_id == t.id
                and self.edges[eid].verb == verb)

        edge = Edge(id=new_id(), source_id=s.id, target_id=t.id, verb=verb)
        self.edges[edge.id] = edge
        self.out_edges.setdefault(s.id, []).append(edge.id)
        self.in_edges.setdefault(t.id, []).append(edge.id)
        self.store.save_edge(edge)
        self._register_in_graph(edge.id, s.id, is_edge=True)
        self._mark_dirty(s.id)
        self._attach_provenance(edge, provenance, source=s, target=t)
        self._note_self_write()
        return edge

    def _attach_provenance(self, edge: Edge, provenance: str | None,
                           *, source: Node, target: Node) -> Edge:
        # Records who asserts the claim made by this triple's target.
        # user_asserts points from the source, agent_inferred creates an
        # agent node. source_says needs nothing extra.
        if not provenance or provenance == "source_says":
            return edge
        asserter = self._resolve("agent", create=True) if \
            provenance == "agent_inferred" else source
        already = any(
            self.edges[eid].target_id == target.id
            and self.edges[eid].verb == provenance
            for eid in self.out_edges.get(asserter.id, []))
        if already:
            return edge
        pe = Edge(id=new_id(), source_id=asserter.id,
                  target_id=target.id, verb=provenance)
        self.edges[pe.id] = pe
        self.out_edges.setdefault(pe.source_id, []).append(pe.id)
        self.in_edges.setdefault(pe.target_id, []).append(pe.id)
        self.store.save_edge(pe)
        self._register_in_graph(pe.id, target.id, is_edge=True)
        return edge

    def create_node(self, label: str, content: str = "", type: str = "fact",
                    under_graph: str | None = None) -> Node:
        gref = under_graph or self.manifest.root_graph_id
        g = self.graphs[gref]
        node = Node(id=new_id(), label=label, type=type, content=content)
        self.nodes[node.id] = node
        self._index_node(node)
        if self._vector_index is not None:
            self._vector_index.ensure_node(node)
            self._vector_index.save()
        g.node_ids.add(node.id)
        self.parent_of[node.id] = (g.id, g.parent_node_id)
        self.store.save_node(node)
        self.store.save_graph(g)
        self._mark_dirty(node.id)
        self._note_self_write()
        return node

    def forget(self, ref: str, *, recursive: bool = True) -> int:
        # Explicit deletion only. Nothing in the system calls this on its own.
        # Removal order matters: edges first, then nested contents, then
        # memberships, then caches, then identity, then the file.
        self.refresh_if_changed()
        node = self._resolve(ref)
        # membership change: dirty every fork viewing this tier before
        # the pointers come apart
        self._mark_dirty(node.id)
        removed = 1

        for eid in list(self.out_edges.get(node.id, [])) + \
                   list(self.in_edges.get(node.id, [])):
            self.delete_edge(eid)

        if node.child_graph_id and recursive:
            removed += self.forget_graph(node.child_graph_id)

        entry = self.parent_of.pop(node.id, None)
        if entry and entry[0] in self.graphs:
            self.graphs[entry[0]].node_ids.discard(node.id)
            self.store.save_graph(self.graphs[entry[0]])

        self._fts.pop(node.id, None)
        self.out_edges.pop(node.id, None)
        self.in_edges.pop(node.id, None)
        if self._vector_index is not None:
            self._vector_index.remove(node.id)
            self._vector_index.save()
        del self.nodes[node.id]
        self.store.delete_node(node.id)
        self._dirty_maps.discard(node.id)
        self._note_self_write()
        return removed

    def delete_edge(self, edge_id: str) -> None:
        e = self.edges.pop(edge_id, None)
        if not e:
            return
        if edge_id in self.out_edges.get(e.source_id, []):
            self.out_edges[e.source_id].remove(edge_id)
        if edge_id in self.in_edges.get(e.target_id, []):
            self.in_edges[e.target_id].remove(edge_id)
        self.store.delete_edge(edge_id)
        g = self._graph_containing_edge(e)
        if g:
            g.edge_ids.discard(edge_id)
            self.store.save_graph(g)

    def forget_graph(self, graph_id: str) -> int:
        g = self.graphs.get(graph_id)
        if not g:
            return 0
        removed = 0
        for nid in list(g.node_ids):
            removed += self.forget(nid, recursive=False)
        for eid in list(g.edge_ids):
            self.delete_edge(eid)
        del self.graphs[graph_id]
        self.store.delete_graph(graph_id)
        return removed

    # Inspection

    def neighbors(self, ref: str) -> list[dict]:
        n = self._resolve(ref)
        out = []
        for eid in self.out_edges.get(n.id, []):
            e = self.edges[eid]
            out.append({"verb": e.verb, "direction": "out",
                        "other": self.nodes[e.target_id].label})
        for eid in self.in_edges.get(n.id, []):
            e = self.edges[eid]
            out.append({"verb": e.verb, "direction": "in",
                        "other": self.nodes[e.source_id].label})
        return out

    def subtree(self, graph_ref: str) -> dict:
        g = self._resolve_graph(graph_ref)
        parent_label = (self.nodes[g.parent_node_id].label
                        if g.parent_node_id else None)
        return {
            "graph": g.name,
            "parent_node": parent_label,
            "nodes": sorted(self.nodes[n].label
                            for n in g.node_ids if n in self.nodes),
        }

    def provenance_report(self) -> dict:
        # Every claim node should trace back to a provenance edge. This
        # audit lists the ones that do not.
        from .epistemics import PROVENANCE_VERBS

        unbacked, backed = [], 0
        for n in self.nodes.values():
            if n.type == "anchor":
                continue
            has = any(self.edges[eid].verb in PROVENANCE_VERBS
                      for eid in self.in_edges.get(n.id, []))
            if has:
                backed += 1
            else:
                unbacked.append({"id": n.id, "label": n.label})
        return {"backed": backed, "unbacked": unbacked}

    # Write placement

    def placement_hint(self, source_ref: str, target_ref: str) -> dict:
        # As deep as true, no deeper. Cross-branch triples generalize at
        # the shared ancestor. Same-subtree triples file at leaf level.
        ctx = self.lca_context([source_ref, target_ref])
        if ctx["lca"] is None:
            return {"suggest": "root", "reason": "no shared context"}
        if ctx["lca_kind"] == "node":
            return {
                "suggest": f"under [{ctx['lca_name']}]",
                "reason": "cross-branch triple, generalize at the shared "
                          "ancestor rather than leaf level",
            }
        return {
            "suggest": f"inside graph '{ctx['lca_name']}'",
            "reason": "same-domain triple, file at leaf level",
        }

    # Promotion split

    def fanout(self, node_id: str) -> int:
        # How crowded this node's neighborhood is. High fanout means the
        # parent swallowed too much and should be split.
        n = self.nodes[node_id]
        gid = n.child_graph_id or (
            self.parent_of.get(node_id, (None,))[0])
        g = self.graphs.get(gid) if gid else None
        if not g:
            return 0
        return sum(1 for nid in g.node_ids
                   if nid != node_id and nid in self.nodes)

    def _group_members(self, members: list[Node], strategy: str,
                       threshold: float) -> list[tuple[str, list[Node]]]:
        # Partitions tier members for a promotion-split into
        # (label hint, [members]) pairs. "verbs" keys on the sorted verb
        # neighborhood; "semantic" clusters greedily by embedding cosine
        # against each cluster's leader (deterministic: members walk in
        # id order), so content similarity decides the boundary and
        # same-verb strangers stay apart. Falls back to verbs when no
        # embedder is loaded.
        if strategy == "semantic" and self._vector_index is not None:
            from .semantic import cosine

            self._vector_index.ensure_all(self.nodes)
            clusters: list[list[Node]] = []
            for m in sorted(members, key=lambda n: n.id):
                v = self._vector_index.vectors.get(m.id)
                best: list[Node] | None = None
                best_sim = threshold
                if v:
                    for cl in clusters:
                        lv = self._vector_index.vectors.get(cl[0].id)
                        if lv:
                            sim = cosine(v, lv)
                            if sim >= best_sim:
                                best, best_sim = cl, sim
                if best is None:
                    clusters.append([m])
                else:
                    best.append(m)
            return [(cl[0].label[:40], cl) for cl in clusters]
        groups: dict[tuple, list[Node]] = {}
        for m in members:
            verbs = tuple(sorted(
                {self.edges[e].verb
                 for e in self.out_edges.get(m.id, [])} |
                {self.edges[e].verb
                 for e in self.in_edges.get(m.id, [])}))
            groups.setdefault(verbs or ("unlinked",), []).append(m)
        return [(("related" if not verbs else
                  ", ".join(verbs[:3])[:40]), ms)
                for verbs, ms in groups.items()]

    def split_if_overloaded(self, node_id: str, cap: int = 9,
                            strategy: str = "verbs",
                            threshold: float = 0.5) -> list[str]:
        # Groups tier members into intermediate anchor nodes: "verbs"
        # keys on the shared verb neighborhood (pure structure, no
        # model), "semantic" clusters by embedding cosine so content
        # similarity decides where the boundary goes. Returns the
        # created group node ids.
        parent = self.nodes[node_id]
        gid = parent.child_graph_id or self.parent_of.get(
            node_id, (None, None))[0] or self.manifest.root_graph_id
        g = self.graphs[gid]

        members = [self.nodes[nid] for nid in g.node_ids
                   if nid != node_id and nid in self.nodes]
        if len(members) <= cap:
            return []

        groups = self._group_members(members, strategy, threshold)
        oversized = [hint for hint, ms in groups if len(ms) > cap]
        if len(groups) < 2 and not oversized:
            return []  # no boundary to cut along, refuse

        created: list[str] = []
        for hint, ms in groups:
            if len(ms) < 2:
                continue  # wrapping singletons adds nothing
            label = f"{parent.label}: {hint}"
            grp = Node(id=new_id(), label=label, type="anchor")
            self.nodes[grp.id] = grp
            self._index_node(grp)
            g.node_ids.add(grp.id)
            # anchors chain up through the owning graph's parent node, the
            # same shape _load() rebuilds, so lca walks survive the session
            self.parent_of[grp.id] = (g.id, g.parent_node_id)
            self.store.save_node(grp)

            sub = Graph(id=new_id(), name=label, parent_node_id=grp.id)
            self.graphs[sub.id] = sub
            for m in ms:
                g.node_ids.discard(m.id)
                sub.node_ids.add(m.id)
                self.parent_of[m.id] = (sub.id, grp.id)
            self.store.save_graph(sub)

            grp.child_graph_id = sub.id
            self.store.save_node(grp)
            created.append(grp.id)

        self.store.save_graph(g)
        self._mark_dirty(node_id)
        # split wrote files: refresh the fingerprint so the next read
        # does not mistake our own writes for an external editor and
        # wipe the dirty marks we just set
        self._note_self_write()
        return created
