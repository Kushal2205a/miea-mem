"""Semantic layer: embeddings as derived cache — never truth.

Activated when scale/paraphrase-recall demands it (see DESIGN.md Design
Principle 2: "No embeddings. Not yet." — 'yet' arrived the first time an
agent failed to match 'food' ↔ 'Biryani').

Design constraints:
- Vectors live in a sidecar index (<workspace>/.index/vectors.json), never in
  the entity files; rebuildable at any time from nodes.
- Local embedding model, no API calls.
- Graceful degradation: without a model installed, Memory.search() behaves
  exactly as before (pure FTS).
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from .model import Node


class Embedder(Protocol):
    """Minimal embedding interface; any local model can satisfy this."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class NullEmbedder:
    """Always unavailable — keeps Memory fully offline-capable."""

    dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no embedder available")


def try_load_embedder() -> Embedder | None:
    """Find an installed local embedding backend; None if absent.

    Preference order: sentence-transformers (nomic / minilm) — heavy dep,
    kept opt-in via [project.optional-dependencies] semantic.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return None

    for model_name in ("nomic-ai/nomic-embed-text-v1.5",):
        try:
            model = SentenceTransformer(model_name)
            return _STEmbedder(model)
        except Exception:
            continue
    # fall back to whatever small default is cached locally
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return _STEmbedder(model)
    except Exception:
        return None


class _STEmbedder:
    def __init__(self, model: Any):
        self._model = model
        self.dim = model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def node_text(n: Node) -> str:
    """What gets embedded for a node — label weighted by repetition."""
    parts = [n.label, n.label, n.type, " ".join(n.tags)]
    if n.content:
        parts.append(n.content[:500])
    return "\n".join(p for p in parts if p)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorIndex:
    """Sidecar store: node_id → vector. Rebuildable, disposable."""

    def __init__(self, workspace_root: Path, embedder: Embedder):
        self.embedder = embedder
        self.path = Path(workspace_root) / ".index" / "vectors.json"
        self.vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("dim") == self.embedder.dim:
                self.vectors = data["vectors"]

    def save(self) -> None:
        if not self.vectors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"dim": self.embedder.dim, "vectors": self.vectors}))
        tmp.replace(self.path)

    def ensure_node(self, node: Node) -> None:
        """Embed node if missing/stale. Cheap check: presence only."""
        if node.id not in self.vectors:
            self.vectors[node.id] = self.embedder.embed(
                [node_text(node)])[0]

    def ensure_all(self, nodes: dict[str, Node]) -> None:
        missing = [n for nid, n in nodes.items() if nid not in self.vectors]
        if missing:
            vecs = self.embedder.embed([node_text(n) for n in missing])
            for n, v in zip(missing, vecs):
                self.vectors[n.id] = v
            self.save()

    def remove(self, node_id: str) -> None:
        self.vectors.pop(node_id, None)

    def query(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (node_id, cosine) for a query string."""
        if not self.vectors:
            return []
        qv = self.embedder.embed([text])[0]
        scored = [
            (nid, cosine(qv, vec)) for nid, vec in self.vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def rrf_fuse(rank_lists: list[list[str]], k: int = 60,
             top: int = 10) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across result lists (id-ordered best-first)."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for pos, nid in enumerate(ranks):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + pos + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top]
