"""The memory core: flat property-graph + derived indexes + read/write ops.

This is the waterslide. JSON files are truth; this structure is what the
agent actually talks to through shaped payloads.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .model import Breadth, Edge, Graph, Manifest, Node, new_id
from .store import Store

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its of on or "
    "that the this to was were will with".split()
)

# Signpost size: how many destinations a parent shows at once (Miller-ish).
TOP_K = 7


def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9_]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def _parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Payloads (shaped for the LLM — never raw dumps)
# ---------------------------------------------------------------------------


@dataclass
class Destination:
    """One entry on a signpost: a reference, not a summary."""

    node_id: str
    label: str
    verb: str | None  # edge verb if reached via an edge; None for children
    direction: str | None  # "out" | "in" | "child"
    score: float = 0.0
    epistemic_status: str | None = None  # shown inline when noteworthy

    def render(self) -> str:
        via = f" --{self.verb}--> " if self.direction == "out" else (
            f" <--{self.verb}-- " if self.direction == "in" else " ⊃ "
        )
        line = f"[{self.label}]{via}(score {self.score:.2f}"
        if self.epistemic_status and self.epistemic_status not in (
                "unverifiable",):
            line += f" · {self.epistemic_status}"
        return line + ")"


@dataclass
class Payload:
    """Rideable payload: where you are + where you can go."""

    node: Node
    path_so_far: list[str] = field(default_factory=list)  # sentence fragments
    signpost: list[Destination] = field(default_factory=list)
    total_destinations: int = 0  # n, so the agent knows pages exist beyond k
    page: int = 0                # current signpost page (0-based, TOP_K per page)
    matched_edges: list[dict] = field(default_factory=list)  # write-tier extras

    def render(self) -> str:
        lines = [f"● [{self.node.label}] ({self.node.type})"]
        if self.node.content:
            lines.append(f"  {self.node.content}")
        if self.path_so_far:
            lines.append(f"  sentence-so-far: {' '.join(self.path_so_far)}")
        if self.node.child_graph_id:
            lines.append("  (has nested graph)")
        lines.append(f"  epistemic: {self.node.epistemic_status}")
        if self.signpost:
            lines.append("  destinations:")
            for d in self.signpost:
                lines.append(f"    {d.render()}")
            if self.total_destinations > len(self.signpost):
                lines.append(
                    f"    … {self.total_destinations - len(self.signpost)} more "
                    "(scoped query to reach them)"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class Memory:
    """In-memory graph loaded from the workspace. All ops go through here."""

    def __init__(self, root: str):
        self.store = Store(root)
        self.manifest: Manifest = self.store.load_manifest()

        # Flat tables — the actual data structure
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.graphs: dict[str, Graph] = {}

        # Derived indexes (rebuildable)
        self.out_edges: dict[str, list[str]] = {}   # node_id -> [edge_id]
        self.in_edges: dict[str, list[str]] = {}
        self.parent_of: dict[str, tuple[str, str | None]] = {}  # node_id -> (graph_id, parent_node_id|None)
        self._fts: dict[str, Counter] = {}          # node_id -> token counts
        self._dirty_maps: set[str] = set()          # ancestors needing map regen

        self._load()

    # -- loading -------------------------------------------------------------

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
        # containment overlay
        for gid, g in self.graphs.items():
            for nid in g.node_ids:
                self.parent_of[nid] = (gid, g.parent_node_id)

    def _index_node(self, n: Node) -> None:
        text = " ".join([n.label, n.type, " ".join(n.tags), n.content])
        self._fts[n.id] = Counter(_tokenize(text))

    def reindex(self) -> None:
        """Rebuild all derived indexes from the flat tables."""
        self._fts.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self.parent_of.clear()
        for n in self.nodes.values():
            self._index_node(n)
        for e in self.edges.values():
            self.out_edges.setdefault(e.source_id, []).append(e.id)
            self.in_edges.setdefault(e.target_id, []).append(e.id)
        for gid, g in self.graphs.items():
            for nid in g.node_ids:
                self.parent_of[nid] = (gid, g.parent_node_id)
        self._dirty_maps = set(m[1] for m in self.parent_of.values() if m[1])

    # -- ranking -------------------------------------------------------------

    def breadth_score(self, node_id: str, now: float | None = None) -> float:
        """breadth × recency decay. Order-only signal."""
        n = self.nodes[node_id]
        now = now or datetime.now(timezone.utc).timestamp()
        access = n.breadth.access_count + n.breadth.traversal_count
        recency_boost = 0.0
        ts = _parse_iso(n.breadth.last_accessed) or _parse_iso(n.created_at)
        if ts:
            age_days = max(0.0, (now - ts) / 86400)
            recency_boost = math.exp(-age_days / 14.0) * 2.0  # fresh ≈ up to +2
        return math.log1p(access) + recency_boost

    # -- search / entry points ----------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[tuple[Node, float]]:
        q = Counter(_tokenize(query))
        if not q:
            return []
        scored: list[tuple[float, str]] = []
        for nid, tokens in self._fts.items():
            overlap = sum((tokens & q).values())
            if not overlap:
                continue
            norm = sum(tokens.values()) or 1
            scored.append((overlap / math.sqrt(norm), nid))
        scored.sort(reverse=True)
        out = []
        for _, nid in scored[:limit]:
            out.append((self.nodes[nid], self.breadth_score(nid)))
        return out

    # -- read loop -----------------------------------------------------------

    def land(
        self,
        ref: str,
        *,
        include_edges: bool = False,
        mark_access: bool = True,
        page: int = 0,
    ) -> Payload:
        """Land on a node by id or unique-ish label. Returns rideable payload.

        `page` walks the signpost beyond top-k when fan-out is large
        (page 1, 2, …). The payload reports total destinations so the agent
        knows more pages exist; query_scoped remains the mediated shortcut.
        """
        node = self._resolve(ref)
        if mark_access:
            from .model import now_iso

            node.breadth.access_count += 1
            node.breadth.last_accessed = now_iso()
            self.store.save_node(node)
        return self._payload(node, include_edges=include_edges, page=page)

    def steer(
        self,
        ref: str,
        destination_label_or_id: str,
        *,
        include_edges: bool = False,
    ) -> Payload:
        """Take one branch of the slide toward a signpost destination."""
        node = self._resolve(ref)
        dest = self._find_destination(node, destination_label_or_id)
        if dest is None:
            raise LookupError(f"no destination matching {destination_label_or_id!r}")
        # traversal counting on both ends + the connecting edge
        nxt = self.nodes[dest.node_id]
        for n in (node, nxt):
            n.breadth.traversal_count += 1
            from .model import now_iso
            n.breadth.last_accessed = now_iso()
            self.store.save_node(n)
        payload = self._payload(nxt, include_edges=include_edges)
        payload.path_so_far = list(node_path_sentence(self, node)) 
        if dest.verb:
            payload.path_so_far.append(f"--{dest.verb}-->")
        payload.path_so_far.append(f"[{nxt.label}]")
        return payload

    def query_scoped(self, graph_ref: str, query: str, limit: int = 5) -> list[Payload]:
        """Mediated deep dive inside one subtree. Returns answers, not paths."""
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
        return out

    def lca_context(self, refs: list[str]) -> dict:
        """Lowest common ancestor of nodes: minimal coherent context.

        Pure graph math over parent/containment pointers. Chains walk
        node → its graph → graph's parent node → ... Two nodes in the same
        graph share that graph as common ground.
        """
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
                    break  # reached a root-level graph
            return steps

        chains = [chain(n.id) for n in resolved]
        common = None
        # direct ancestry: if node A appears in B's chain, A is the LCA
        for i, n in enumerate(resolved):
            if any(n.id in chains[j] for j in range(len(chains)) if j != i):
                common = n.id
                break
        if common is None:
            # chains run leaf→root; walk from the root end while they agree.
            # Divergence point (or shared root graph) is the LCA.
            for a, b in zip(reversed(chains[0]),
                            reversed(chains[1] if len(chains) > 1 else [])):
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

    # -- write tier ----------------------------------------------------------

    def write_triple(self, source_ref: str, verb: str, target_ref: str,
                     *, create_missing: bool = False) -> Edge:
        """Add a named edge (noun-verb-noun). Dedup: refuse exact duplicate.

        System verbs (corroborated_by, contradicted_by, …) are reserved for
        the epistemic verifier — user/agent writes can never create or flip
        corroboration. That's the manipulation-resistance guarantee.
        """
        from .epistemics import SYSTEM_VERBS

        if verb in SYSTEM_VERBS:
            raise ValueError(
                f"verb {verb!r} is system-reserved (epistemic verifier only)")
        s = self._resolve(source_ref, create=create_missing)
        t = self._resolve(target_ref, create=create_missing)
        for eid in self.out_edges.get(s.id, []):
            e = self.edges[eid]
            if e.target_id == t.id and e.verb == verb:
                return e  # already known — idempotent
        edge = Edge(id=new_id(), source_id=s.id, target_id=t.id, verb=verb)
        self.edges[edge.id] = edge
        self.out_edges.setdefault(s.id, []).append(edge.id)
        self.in_edges.setdefault(t.id, []).append(edge.id)
        self.store.save_edge(edge)
        self._register_in_graph(edge.id, s.id, is_edge=True)
        self._mark_dirty(s.id)
        return edge

    def create_node(self, label: str, content: str = "", type: str = "fact",
                    under_graph: str | None = None) -> Node:
        gref = under_graph or self.manifest.root_graph_id
        g = self.graphs[gref]
        node = Node(id=new_id(), label=label, type=type, content=content)
        self.nodes[node.id] = node
        self._index_node(node)
        g.node_ids.add(node.id)
        self.parent_of[node.id] = (g.id, g.parent_node_id)
        self.store.save_node(node)
        self.store.save_graph(g)
        self._mark_dirty(node.id)
        return node

    def forget(self, ref: str, *, recursive: bool = True) -> int:
        """Explicit deletion only. Never automatic."""
        node = self._resolve(ref)
        removed = 1
        # edges first
        for eid in list(self.out_edges.get(node.id, [])) + \
                   list(self.in_edges.get(node.id, [])):
            self.delete_edge(eid)
        if node.child_graph_id and recursive:
            removed += self.forget_graph(node.child_graph_id)
        # containment bookkeeping
        entry = self.parent_of.pop(node.id, None)
        if entry and entry[0] in self.graphs:
            self.graphs[entry[0]].node_ids.discard(node.id)
            self.store.save_graph(self.graphs[entry[0]])
        self._fts.pop(node.id, None)
        self.out_edges.pop(node.id, None)
        self.in_edges.pop(node.id, None)
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
            removed += 0
        del self.graphs[graph_id]
        self.store.delete_graph(graph_id)
        return removed

    # -- inspection (write tier) ----------------------------------------------

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
        return {
            "graph": g.name,
            "parent_node": self.nodes[g.parent_node_id].label if g.parent_node_id else None,
            "nodes": sorted(self.nodes[n].label for n in g.node_ids if n in self.nodes),
        }

    def provenance_report(self) -> dict:
        """Audit: every claim node should trace to a provenance edge
        (user_asserts / agent_inferred / source_says)."""
        from .epistemics import PROVENANCE_VERBS

        unbacked, backed = [], 0
        for n in self.nodes.values():
            if n.type == "anchor":
                continue  # anchors are structural (incl. system source nodes)
            has = False
            for eid in self.in_edges.get(n.id, []):
                if self.edges[eid].verb in PROVENANCE_VERBS:
                    has = True
                    break
            if has:
                backed += 1
            else:
                unbacked.append({"id": n.id, "label": n.label})
        return {"backed": backed, "unbacked": unbacked}

    # -- write placement -------------------------------------------------------

    def placement_hint(self, source_ref: str, target_ref: str) -> dict:
        """'As deep as it's true, no deeper.'

        Structural heuristic (no LLM): if source and target already live in
        different subtrees, the triple generalizes across domains → suggest
        the LCA level (an intermediate node), not a leaf. Same subtree → file
        at leaf level.
        """
        ctx = self.lca_context([source_ref, target_ref])
        if ctx["lca"] is None:
            return {"suggest": "root", "reason": "no shared context"}
        kind = ctx["lca_kind"]
        if kind == "node":
            return {
                "suggest": f"under [{ctx['lca_name']}]",
                "reason": "cross-branch triple — generalize at the shared "
                          "ancestor, not at leaf level",
            }
        return {
            "suggest": f"inside graph '{ctx['lca_name']}'",
            "reason": "same-domain triple — file at leaf level",
        }

    # -- promotion-split ------------------------------------------------------

    def fanout(self, node_id: str) -> int:
        """Leaf pressure gauge: distinct child-graph members under this node."""
        n = self.nodes[node_id]
        gid = n.child_graph_id or (
            self.parent_of.get(node_id, (None,))[0])
        g = self.graphs.get(gid) if gid else None
        if not g:
            return 0
        return sum(1 for nid in g.node_ids
                   if nid != node_id and nid in self.nodes)

    def split_if_overloaded(self, node_id: str,
                            cap: int = 9) -> list[str]:
        """Promotion-split when fan-out exceeds the cap.

        Groups children by shared verb-neighborhood into new intermediate
        siblings (pure structure — no LLM, no embeddings): children that send
        or receive edges with the same verbs get promoted together. The tree
        grows a level instead of widening. Returns created group-node ids.
        """
        parent = self.nodes[node_id]
        gid = parent.child_graph_id or self.parent_of.get(
            node_id, (None, None))[0] or self.manifest.root_graph_id
        g = self.graphs[gid]
        members = [self.nodes[nid] for nid in g.node_ids
                   if nid != node_id and nid in self.nodes]
        if len(members) <= cap:
            return []

        # signature = sorted set of verbs touching each member
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
            return []  # no structural signal to cluster on; refuse to split

        created: list[str] = []
        for verbs, ms in groups.items():
            if len(ms) < 2:
                continue  # don't promote singletons
            label = f"{parent.label}: " + (", ".join(verbs[:3])[:40] or "related")
            grp = Node(id=new_id(), label=label, type="anchor")
            self.nodes[grp.id] = grp
            self._index_node(grp)
            g.node_ids.add(grp.id)
            self.parent_of[grp.id] = (g.id, None)
            self.store.save_node(grp)
            # re-parent members into a fresh sub-graph owned by the group node
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

    # -- internals -----------------------------------------------------------

    def _payload(self, node: Node, *, include_edges: bool = False,
                 page: int = 0) -> Payload:
        dests: list[Destination] = []
        # edge neighbors (the slide walls)
        for eid in self.out_edges.get(node.id, []):
            e = self.edges[eid]
            other = self.nodes.get(e.target_id)
            if other:
                dests.append(Destination(other.id, other.label, e.verb, "out",
                                         self.breadth_score(other.id),
                                         other.epistemic_status))
        for eid in self.in_edges.get(node.id, []):
            e = self.edges[eid]
            other = self.nodes.get(e.source_id)
            if other:
                dests.append(Destination(other.id, other.label, e.verb, "in",
                                         self.breadth_score(other.id),
                                         other.epistemic_status))
        # containment: sibling leaves under our child graph, or our siblings
        child_gid = node.child_graph_id or (
            self.parent_of.get(node.id, (None,))[0]
        )
        g = self.graphs.get(child_gid) if child_gid else None
        if g:
            for nid in g.node_ids:
                if nid == node.id or nid not in self.nodes:
                    continue
                dests.append(Destination(nid, self.nodes[nid].label, None, "child",
                                         self.breadth_score(nid)))
        # order-only ranking: breadth × recency, spread across divergent dests
        seen: set[str] = set()
        deduped: list[Destination] = []
        for d in dests:  # prefer edge-view over child-view of same node
            key = d.node_id  # one destination per node; edge-view kept over child-view
            if key not in seen:
                seen.add(key)
                deduped.append(d)
        deduped.sort(key=lambda d: d.score, reverse=True)
        total = len(deduped)
        start = page * TOP_K
        payload = Payload(node=node, signpost=deduped[start:start + TOP_K],
                          total_destinations=total)
        payload.page = page
        if include_edges:
            payload.matched_edges = [
                {"id": eid, "verb": self.edges[eid].verb}
                for eid in self.out_edges.get(node.id, [])
            ]
        return payload

    def _resolve(self, ref: str, *, create: bool = False) -> Node:
        if ref in self.nodes:
            return self.nodes[ref]
        matches = [n for n in self.nodes.values() if n.label.lower() == ref.lower()]
        if len(matches) == 1:
            return matches[0]
        if create:
            return self.create_node(label=ref)
        raise LookupError(f"no node {ref!r}")

    def _resolve_graph(self, ref: str) -> Graph:
        if ref in self.graphs:
            return self.graphs[ref]
        matches = [g for g in self.graphs.values() if g.name.lower() == ref.lower()]
        if len(matches) == 1:
            return matches[0]
        raise LookupError(f"no graph {ref!r}")

    def _find_destination(self, node: Node, want: str) -> Destination | None:
        p = self._payload(node)
        for d in p.signpost:
            if d.node_id == want or d.label.lower() == want.lower():
                return d
        # allow steering to any destination even beyond top-k page
        for d in self._all_destinations(node):
            if d.node_id == want or d.label.lower() == want.lower():
                return d
        return None

    def _all_destinations(self, node: Node) -> list[Destination]:
        full = self._payload(node)
        # cheap way to get everything: temporarily bypass top-k
        saved = TOP_K
        try:
            globals()["TOP_K"] = 10**9
            full = self._payload(node)
        finally:
            globals()["TOP_K"] = saved
        return full.signpost

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
        """Mark ancestors so their divergence maps regenerate on next access."""
        entry = self.parent_of.get(node_id)
        while entry:
            gid, pid = entry
            if pid:
                self._dirty_maps.add(pid)
                entry = self.parent_of.get(pid)
            else:
                break


def node_path_sentence(mem: Memory, node: Node) -> list[str]:
    """Sentence-so-far for landing: just the node itself."""
    return [f"[{node.label}]"]
