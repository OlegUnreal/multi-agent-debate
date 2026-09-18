"""Deterministic, offline text embeddings.

Strategy (in order of preference):

1. ``TfidfVectorizer`` over word (1,2)-grams and char_wb (2,4)-grams,
   hstacked, reduced with ``TruncatedSVD`` and L2-normalised to float32.
2. Signed ``blake2b`` feature hashing when the corpus is too small (or too
   narrow) for a meaningful SVD — the "tiny corpora" fallback.

No model downloads, no network: the same inputs always produce bit-identical
vectors, which is what lets the whole test suite run offline.
"""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _signed_hash_vector(text: str, dim: int) -> np.ndarray:
    """Count-min style signed hashing of tokens and their char trigrams.

    Each feature is mapped to one bucket with a +/- sign taken from a spare
    hash bit, so collisions cancel in expectation instead of compounding.
    """
    vec = np.zeros(dim, dtype=np.float64)
    words = _tokens(text)
    features: list[str] = list(words)
    for tok in words:
        features.extend(tok[i:i + 3] for i in range(max(len(tok) - 2, 0)))
    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        code = int.from_bytes(digest, "big")
        bucket = code % dim
        sign = 1.0 if (code >> 63) & 1 else -1.0
        vec[bucket] += sign
    return vec


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # keep zero vectors zero without dividing
    out = mat / norms
    return np.ascontiguousarray(out, dtype=np.float32)


class TextEmbedder:
    """Fit-on-corpus TF-IDF+SVD embedder with a hashing fallback.

    ``fit`` decides once per corpus which backend to use; ``transform`` then
    maps any texts into the fitted space. ``min_docs_for_svd`` guards the
    degenerate regime (one or two documents) where SVD directions are noise.
    """

    def __init__(self, dim: int = 96, min_docs_for_svd: int = 4) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = dim
        self.min_docs_for_svd = min_docs_for_svd
        self.backend: str | None = None
        self.out_dim: int = dim
        self._vectorizer = None
        self._svd = None

    # -- fitting ---------------------------------------------------------

    def fit(self, corpus: Sequence[str]) -> "TextEmbedder":
        from scipy.sparse import hstack as _hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [c if isinstance(c, str) else str(c) for c in corpus]
        self.backend = None
        if len(corpus) >= self.min_docs_for_svd and any(c.strip() for c in corpus):
            try:
                word = TfidfVectorizer(
                    lowercase=True, sublinear_tf=True,
                    ngram_range=(1, 2), min_df=1, norm="l2",
                )
                char = TfidfVectorizer(
                    analyzer="char_wb", lowercase=True, sublinear_tf=True,
                    ngram_range=(2, 4), min_df=1, norm="l2",
                )
                xw = word.fit_transform(corpus)
                xc = char.fit_transform(corpus)
                x = _hstack([xw, xc])
                # identical docs -> zero variance -> TruncatedSVD divides by 0;
                # computed via E[x^2]-E[x]^2 because sparse has no .var()
                mean_col = np.asarray(x.mean(axis=0)).ravel()
                sq_mean = np.asarray(x.multiply(x).mean(axis=0)).ravel()
                total_var = float(np.sum(np.maximum(sq_mean - mean_col ** 2, 0.0)))
                n_components = int(min(self.dim, x.shape[1] - 1, x.shape[0] - 1))
                if (total_var > 1e-12 and n_components >= 2
                        and x.shape[0] >= self.min_docs_for_svd):
                    svd = TruncatedSVD(
                        n_components=n_components,
                        random_state=0,  # deterministic projection
                    )
                    svd.fit(x)
                    self._vectorizer = (word, char)
                    self._svd = svd
                    self.out_dim = n_components
                    self.backend = "tfidf_svd"
                    return self
            except Exception:  # noqa: BLE001 — any sklearn edge case -> hashing
                pass
        self._vectorizer = None
        self._svd = None
        self.out_dim = self.dim
        self.backend = "hash"
        return self

    # -- transforming ----------------------------------------------------

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend is None:
            self.fit(texts)
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if self.backend == "tfidf_svd":
            word, char = self._vectorizer
            from scipy.sparse import hstack as _hstack
            x = _hstack([word.transform(texts), char.transform(texts)])
            mat = self._svd.transform(x)
        else:
            mat = np.vstack([_signed_hash_vector(t, self.out_dim) for t in texts]) \
                if texts else np.zeros((0, self.out_dim))
        if mat.shape[0] == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        return _l2_normalise(np.asarray(mat, dtype=np.float64))

    def embed(self, text: str) -> np.ndarray:
        """Convenience: one text -> one vector (must be fitted)."""
        return self.transform([text])[0]


def embed_with_shared_space(query: str, corpus: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on ``corpus + [query]`` and return (query_vec, corpus_matrix).

    Used whenever query and stored texts were not embedded together before:
    TF-IDF+SVD is corpus-dependent, so re-fitting is the only way to keep
    vectors comparable. Runs are deterministic.
    """
    emb = TextEmbedder().fit(list(corpus) + [query])
    matrix = emb.transform(list(corpus) + [query])
    return matrix[-1], matrix[:-1]
