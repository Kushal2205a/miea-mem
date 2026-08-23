"""Epistemics: async annotation, never admission control.

Claims land as unverified; a pluggable verifier (SERP-level by design) annotates
them afterward. The system only ever ADDS edges/status — it never deletes or
refuses. Users write provenance; only the verifier writes corroboration.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from .model import Edge, Node, new_id, now_iso

# Status values a claim node may carry.
UNVERIFIED = "unverified"
CORROBORATED = "corroborated"
CONTRADICTED = "contradicted"
CONTESTED = "contested"
UNVERIFIABLE = "unverifiable"

# Verbs reserved for the system — user/agent writes can never create these.
CORROBORATED_BY = "corroborated_by"
CONTRADICTED_BY = "contradicted_by"
SOME_SOURCES_SAY = "some_sources_say"
OTHER_SOURCES_SAY = "other_sources_say"

SYSTEM_VERBS = frozenset({
    CORROBORATED_BY, CONTRADICTED_BY, SOME_SOURCES_SAY, OTHER_SOURCES_SAY,
})

PROVENANCE_VERBS = frozenset({"user_asserts", "agent_inferred", "source_says"})

# Heuristic: claims about the checkable world vs. user-domain statements.
_WORLD_HINTS = re.compile(
    r"\b(is|are|was|were|does|do|can|cannot|causes?|boosts?|improves?|"
    r"prevents?|proven|fact|actually|always|never)\b", re.IGNORECASE,
)


def classify_claim(text: str) -> str:
    """Cheap structural classification — no LLM.

    Returns 'world' (lookupable), 'user' (user-domain: never verified), or
    'opaque' (nothing to look up).
    """
    if not text or len(text.split()) < 3:
        return "opaque"
    if _WORLD_HINTS.search(text):
        return "world"
    return "opaque"


@dataclass
class Verdict:
    status: str                 # CORROBORATED / CONTRADICTED / CONTESTED / UNVERIFIABLE
    sources: list[str]          # short source descriptors from the SERP


class Verifier(ABC):
    """Pluggable lookup backend. Default impl hits a SERP; tests stub this."""

    @abstractmethod
    def verify(self, claim: str) -> Verdict: ...


class NullVerifier(Verifier):
    """Treats everything as unverifiable — offline default."""

    def verify(self, claim: str) -> Verdict:
        return Verdict(status=UNVERIFIABLE, sources=[])


def make_serp_verifier(search_fn) -> Verifier:
    """Wrap a search_fn(query) -> list[result-dicts] into a Verifier.

    The search engine's own verdict is the signal (ranked results, snippets,
    knowledge panels) — we never fetch site content. Agreement heuristic:
    compare negation-bearing top results against the claim's polarity.
    """

    class SerpVerifier(Verifier):
        def verify(self, claim: str) -> Verdict:
            try:
                results = search_fn(claim)
            except Exception:
                return Verdict(status=CONTESTED, sources=[])
            if not results:
                return Verdict(status=UNVERIFIABLE, sources=[])
            titles = [r.get("title", "") for r in results[:5]]
            claim_negated = bool(re.search(
                r"\b(not|no|never|cannot|isn't|aren't|doesn't|don't)\b",
                claim, re.IGNORECASE))
            result_negated = sum(
                1 for t in titles
                if re.search(r"\b(not|no|never|myth|false|misconception|"
                             r"debunk|wrong|actually)\b", t, re.IGNORECASE))
            # >half of top results dispute → contradicted; any dispute of an
            # undisputed claim → contested; else corroborated.
            if result_negated > len(titles) / 2 and not claim_negated:
                return Verdict(status=CONTRADICTED, sources=titles)
            if result_negated and not claim_negated:
                return Verdict(status=CONTESTED, sources=titles)
            return Verdict(status=CORROBORATED, sources=titles)

    return SerpVerifier()


class EpistemicPass:
    """Runs over unverified world-claims and annotates them."""

    def __init__(self, mem, verifier: Verifier):
        self.mem = mem
        self.verifier = verifier

    def pending(self) -> list[Node]:
        """Unverified, lookupable claim nodes."""
        out = []
        for n in self.mem.nodes.values():
            if n.epistemic_status != UNVERIFIED:
                continue
            if classify_claim(f"{n.label} {n.content}") != "world":
                continue
            out.append(n)
        return out

    def run(self, limit: int | None = None) -> list[dict]:
        """Annotate up to `limit` pending claims. Idempotent per node.

        Never deletes anything: contradiction becomes typed edges pointing at
        system-created source nodes; mixed evidence becomes plural-viewpoint
        edges. Annotations carry an as-of date (truth has a shelf life).
        """
        mem = self.mem
        checked_at = now_iso()
        report = []
        todo = self.pending()
        if limit is not None:
            todo = todo[:limit]
        for node in todo:
            verdict = self.verifier.verify(node.content or node.label)
            entry: dict = {"node": node.id, "status": verdict.status}
            if verdict.status == CORROBORATED:
                node.epistemic_status = CORROBORATED
            elif verdict.status == UNVERIFIABLE:
                node.epistemic_status = UNVERIFIABLE
            elif verdict.sources:
                src_status = CONTRADICTED if verdict.status == CONTRADICTED \
                    else CONTESTED
                node.epistemic_status = src_status
                verb = (CONTRADICTED_BY if verdict.status == CONTRADICTED
                        else SOME_SOURCES_SAY)
                made = []
                for title in verdict.sources:
                    src_node = mem._resolve(f"source: {title}", create=True)
                    src_node.type = "anchor"
                    mem.store.save_node(src_node)
                    edge = Edge(id=new_id(), source_id=node.id,
                                target_id=src_node.id, verb=verb)
                    mem.edges[edge.id] = edge
                    mem.out_edges.setdefault(edge.source_id, []).append(edge.id)
                    mem.in_edges.setdefault(edge.target_id, []).append(edge.id)
                    mem.store.save_edge(edge)
                    made.append(src_node.id)
                if verdict.status == CONTESTED and len(made) >= 2:
                    # balance the record: other_sources_say back-references
                    back = Edge(id=new_id(), source_id=made[-1],
                                target_id=node.id, verb=OTHER_SOURCES_SAY)
                    mem.edges[back.id] = back
                    mem.out_edges.setdefault(back.source_id, []).append(back.id)
                    mem.in_edges.setdefault(back.target_id, []).append(back.id)
                    mem.store.save_edge(back)
                entry["source_nodes"] = made
            node.updated_at = checked_at
            node.breadth.last_accessed = checked_at
            mem.store.save_node(node)
            report.append(entry)
        return report


__all__ = [
    "CONTESTED", "CONTRADICTED", "CORROBORATED", "EpistemicPass",
    "NullVerifier", "PROVENANCE_VERBS", "SYSTEM_VERBS", "UNVERIFIABLE",
    "UNVERIFIED", "Verifier", "classify_claim", "make_serp_verifier",
]
