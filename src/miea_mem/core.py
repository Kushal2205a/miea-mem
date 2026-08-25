# Core engine. Loads the workspace files into flat tables, builds derived
# indexes, and implements every operation: search, landing, steering,
# scoped queries, LCA, writes, deletion, splitting.

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .model import Breadth, Edge, Graph, Manifest, Node, new_id
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


_ISO_CACHE: dict[str, float] = {}


def _parse_iso(s: str | None) -> float | None:
    # Parsing dates is hot inside signpost scoring, so parsed values are
    # memoized. Timestamp strings rarely change once written.
    if not s:
        return None
    cached = _ISO_CACHE.get(s)
    if cached is not None:
        return cached
    try:
        ts = datetime.fromisoformat(s).timestamp()
        if len(_ISO_CACHE) > 100_000:
            _ISO_CACHE.clear()
        _ISO_CACHE[s] = ts
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

    def render(self) -> str:
        if self.direction == "out":
            via = f" --{self.verb}--> "
        elif self.direction == "in":
            via = f" <--{self.verb}-- "
        else:
            via = " > "
        line = f"[{self.label}]{via}(score {self.score:.2f}"
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
        self._dirty_maps = set(m[1] for m in self.parent_of.values() if m[1])

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
        ts = _parse_iso(n.breadth.last_accessed) or _parse_iso(n.created_at)
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

        # Siblings through containment: members of our child graph, or of
        # the graph we ourselves live in.
        gid = node.child_graph_id or (
            self.parent_of.get(node.id, (None,))[0]
        )
        g = self.graphs.get(gid) if gid else None
        if g:
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
        # Flags ancestors so their divergence maps regenerate on next access.
        entry = self.parent_of.get(node_id)
        while entry:
            gid, pid = entry
            if pid:
                self._dirty_maps.add(pid)
                entry = self.parent_of.get(pid)
            else:
                break

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
            self._note_self_write()
        return self._payload(node, include_edges=include_edges, page=page)

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
        self._note_self_write()

        payload = self._payload(nxt, include_edges=include_edges)
        payload.path_so_far = [f"[{node.label}]"]
        if dest.verb:
            payload.path_so_far.append(f"--{dest.verb}-->")
        payload.path_so_far.append(f"[{nxt.label}]")
        return payload

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
        return node

    def forget(self, ref: str, *, recursive: bool = True) -> int:
        # Explicit deletion only. Nothing in the system calls this on its own.
        # Removal order matters: edges first, then nested contents, then
        # memberships, then caches, then identity, then the file.
        self.refresh_if_changed()
        node = self._resolve(ref)
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

    def split_if_overloaded(self, node_id: str, cap: int = 9) -> list[str]:
        # Groups children by shared verb neighborhood into intermediate
        # anchor nodes. Pure structure, no model involved. Returns the
        # created group node ids.
        parent = self.nodes[node_id]
        gid = parent.child_graph_id or self.parent_of.get(
            node_id, (None, None))[0] or self.manifest.root_graph_id
        g = self.graphs[gid]

        members = [self.nodes[nid] for nid in g.node_ids
                   if nid != node_id and nid in self.nodes]
        if len(members) <= cap:
            return []

        # Signature is the sorted set of verbs touching each member.
        groups: dict[tuple, list[Node]] = {}
        for m in members:
            verbs = tuple(sorted(
                {self.edges[e].verb
                 for e in self.out_edges.get(m.id, [])} |
                {self.edges[e].verb
                 for e in self.in_edges.get(m.id, [])}))
            groups.setdefault(verbs or ("unlinked",), []).append(m)

        oversized = [v for v, ms in groups.items() if len(ms) > cap]
        if len(groups) < 2 and not oversized:
            return []  # no boundary to cut along, refuse

        created: list[str] = []
        for verbs, ms in groups.items():
            if len(ms) < 2:
                continue  # wrapping singletons adds nothing
            label = f"{parent.label}: " + (", ".join(verbs[:3])[:40]
                                           or "related")
            grp = Node(id=new_id(), label=label, type="anchor")
            self.nodes[grp.id] = grp
            self._index_node(grp)
            g.node_ids.add(grp.id)
            self.parent_of[grp.id] = (g.id, None)
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
        return created
