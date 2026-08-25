"""Semantic search tests: hybrid fusion with a stub embedder, graceful
fallback without one, sidecar lifecycle (create/delete/reindex/gitignore)."""

import math

import pytest

from miea_mem.core import Memory
from miea_mem.semantic import NullEmbedder, VectorIndex, cosine, rrf_fuse
from miea_mem.store import Store


class StubEmbedder:
    """Deterministic toy embedder: hash-based sparse vectors.

    Maps a few seed words to fixed axes so tests can control similarity.
    """

    dim = 8
    AXES = {
        "food": 0, "biryani": 0, "cuisine": 0, "rice": 0,       # food family
        "db": 1, "postgres": 1, "database": 1, "sql": 1,        # db family
        "code": 2, "python": 2,                                 # code family
    }

    def embed(self, texts):
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
    m = Memory(str(root), embedder=StubEmbedder())
    m.create_node("Biryani", content="a rice dish, beloved cuisine")
    m.create_node("Postgres", content="a sql database")
    return m


def test_null_embedder_raises():
    with pytest.raises(RuntimeError):
        NullEmbedder().embed(["x"])


def test_cosine_identical_vs_orthogonal():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_rrf_prefers_items_in_multiple_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a"]])
    assert fused[0][0] == "a" and fused[1][0] == "b"


def test_hybrid_finds_paraphrase(mem: Memory):
    # 'food' shares an axis with biryani's content but NOT its tokens —
    # pure FTS would miss this; the vector leg must surface it.
    hits = mem.search("what food do I like")
    labels = [n.label for n, _ in hits]
    assert "Biryani" in labels


def test_fts_still_wins_for_exact_terms(mem: Memory):
    hits = mem.search("sql database")
    assert hits[0][0].label == "Postgres"


def test_sidecar_lives_in_dotindex_and_persists(mem: Memory, tmp_path):
    root = tmp_path / "ws"
    idx_file = root / ".index" / "vectors.json"
    assert idx_file.exists()
    fresh = Memory(str(root), embedder=StubEmbedder())
    # loaded from disk, not recomputed: vectors present without new writes
    assert fresh._vector_index.vectors


def test_delete_removes_vector(mem: Memory):
    nid = mem._resolve("Biryani").id
    mem.forget("Biryani")
    assert nid not in mem._vector_index.vectors


def test_reindex_rebuilds_vectors(mem: Memory):
    mem.reindex()
    assert len(mem._vector_index.vectors) == len(mem.nodes)


def test_graceful_without_embedder(tmp_path):
    root = tmp_path / "ws"
    Store(root).init_workspace("T")
    m = Memory(str(root))          # default auto to no model installed here?
    # Either vector index is None (no model) or real; either way search works.
    m.create_node("Biryani")
    hits = m.search("biryani")     # exact term must always work
    assert hits and hits[0][0].label == "Biryani"
