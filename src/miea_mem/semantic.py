# Semantic search. Embeds nodes with a pluggable model, stores vectors
# in a sidecar index file, and fuses vector results with keyword ranks.
# Optional: without an embedder the system falls back to keyword search.

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from .model import Node


class Embedder(Protocol):
    # Any object with a dim attribute and an embed method qualifies.
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class NullEmbedder:
    dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no embedder available")


def try_load_embedder() -> Embedder | None:
    # Probe for sentence-transformers. Returns None when absent so callers
    # can degrade to keyword-only search.
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
    # Label appears twice to weight it strongest in the embedding.
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
    # Sidecar store of node_id to vector. Lives in .index/vectors.json,
    # is disposable, and rebuilds from nodes at any time.

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
        # Brute force cosine against every stored vector. Fast enough
        # for thousands of nodes; swap for ANN only if that changes.
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
    # Reciprocal Rank Fusion. Each list votes 1/(k+position) per item;
    # items present in several lists stack their votes. Rank positions
    # are used instead of raw scores because the two methods score on
    # incomparable scales.

    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for pos, nid in enumerate(ranks):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + pos + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top]
