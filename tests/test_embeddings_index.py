"""Tests for deterministic embeddings and the flat/IVF vector index."""
from __future__ import annotations

import numpy as np
import pytest

from debate.embeddings import TextEmbedder, embed_with_shared_space
from debate.vector_index import VectorIndex


CORPUS = [
    "latency regressions appeared after the cache eviction policy changed",
    "the rollback plan restores the previous eviction settings in minutes",
    "OAuth2 token refresh must happen before the access token expires",
    "migrating session storage to redis needs a dual-write cutover period",
    "capacity planning says the current fleet handles twice the peak load",
    "unit tests flake when the clock is not frozen in timezone edge cases",
]


def test_embeddings_are_deterministic_and_unit_norm():
    emb = TextEmbedder(dim=32).fit(CORPUS)
    a = emb.transform(CORPUS)
    b = emb.transform(CORPUS)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert a.shape[1] <= 32


def test_similar_texts_beat_dissimilar():
    emb = TextEmbedder(dim=32).fit(CORPUS)
    v = emb.transform([CORPUS[0], CORPUS[1], CORPUS[2]])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_tiny_corpus_uses_blake2b_hashing_fallback():
    emb = TextEmbedder(dim=32).fit(["one short text"])
    assert emb.backend == "hash"
    vecs = emb.transform(["one short text", "completely different topic about frogs"])
    same = float(vecs[0] @ vecs[0])
    diff = float(vecs[0] @ vecs[1])
    assert same == pytest.approx(1.0, abs=1e-5)
    assert diff < same
    emb2 = TextEmbedder(dim=32).fit(["one short text"])
    np.testing.assert_array_equal(emb2.transform(["one short text"]), vecs[0:1])


def test_empty_text_is_handled():
    emb = TextEmbedder(dim=16).fit(CORPUS)
    v = emb.transform([""])
    assert v.shape == (1, emb.out_dim)
    assert np.isfinite(v).all()


def test_embed_with_shared_space_dims_agree():
    q, docs = embed_with_shared_space("cache eviction latency", CORPUS)
    assert q.shape == docs[0].shape
    assert docs.shape[0] == len(CORPUS)


def test_flat_index_exact_topk():
    emb = TextEmbedder(dim=32).fit(CORPUS)
    idx = VectorIndex(dim=emb.out_dim, ivf_threshold=10_000)
    idx.add(emb.transform(CORPUS))
    hits = idx.search(emb.embed(CORPUS[3]), k=2)
    assert hits[0][0] == 3
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)
    assert idx.mode == "flat"


def test_ivf_mode_engages_and_retrieves_query_itself():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    idx = VectorIndex(dim=16, ivf_threshold=128, n_lists=8, n_probed_lists=8)
    idx.add(base)
    assert idx.mode == "ivf"
    hits = idx.search(base[42], k=1)
    assert hits[0][0] == 42
    # probing every list is exhaustive -> identical to flat scoring
    full = idx.search(base[42], k=5)
    assert len(full) == 5


def test_index_is_deterministic():
    emb = TextEmbedder(dim=32).fit(CORPUS)
    runs = []
    for _ in range(2):
        idx = VectorIndex(dim=emb.out_dim)
        idx.add(emb.transform(CORPUS))
        runs.append(idx.search(emb.embed(CORPUS[0]), k=3))
    assert runs[0] == runs[1]
