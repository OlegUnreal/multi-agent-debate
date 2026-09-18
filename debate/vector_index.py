"""In-process ANN index: exact flat cosine search, IVF for larger runs.

Everything is float32, L2-normalised on insert, so cosine == dot product.
The IVF path uses a deterministically seeded KMeans; candidate ordering is
made total by tie-breaking on insertion id, so two runs never disagree.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Result = tuple[int, float]  # (item id, similarity)


def _as_unit_matrix(vectors: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


class VectorIndex:
    """Flat cosine index that promotes itself to IVF past ``ivf_threshold``.

    All vectors are always kept in memory (the corpus is debate-sized, not
    billion-sized); IVF only changes *how many* of them a query scans.
    ``n_probed_lists`` controls recall/speed; set it to ``n_lists`` to get
    exact results back from the IVF path.
    """

    def __init__(
        self,
        dim: int,
        ivf_threshold: int = 128,
        n_lists: int = 8,
        n_probed_lists: int = 2,
    ) -> None:
        self.dim = dim
        self.ivf_threshold = ivf_threshold
        self.n_lists = n_lists
        self.n_probed_lists = max(1, n_probed_lists)
        self._vectors: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None
        self._assignments: np.ndarray | None = None
        self._trained_n: int = 0

    # -- write path ------------------------------------------------------

    def add(self, vectors: np.ndarray | Sequence[np.ndarray]) -> list[int]:
        matrix = _as_unit_matrix(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {matrix.shape[1]}")
        start = len(self._vectors)
        self._vectors.extend(matrix)
        return list(range(start, start + matrix.shape[0]))

    def __len__(self) -> int:
        return len(self._vectors)

    # -- read path -------------------------------------------------------

    @property
    def mode(self) -> str:
        return "ivf" if len(self._vectors) >= self.ivf_threshold else "flat"

    def _matrix(self) -> np.ndarray:
        return np.vstack(self._vectors) if self._vectors else np.zeros((0, self.dim), np.float32)

    def _train_ivf(self) -> None:
        from sklearn.cluster import KMeans

        matrix = self._matrix()
        n_lists = max(2, min(self.n_lists, len(matrix) // 2))
        kmeans = KMeans(
            n_clusters=n_lists,
            random_state=0,   # deterministic centroids
            n_init=3,
            init="k-means++",
        )
        self._assignments = kmeans.fit_predict(matrix)
        self._centroids = _as_unit_matrix(kmeans.cluster_centers_)
        self._trained_n = len(matrix)

    def search(self, query: np.ndarray, k: int = 5) -> list[Result]:
        if len(self._vectors) == 0 or k <= 0:
            return []
        q = _as_unit_matrix(np.asarray(query, dtype=np.float32))[0]
        if q.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
        if self.mode == "flat":
            scores = self._matrix() @ q
        else:
            if self._centroids is None or self._trained_n != len(self._vectors):
                self._train_ivf()  # retrain whenever the tail grew
            probes = min(self.n_probed_lists, len(self._centroids))
            probe_order = np.argsort(-(self._centroids @ q), kind="stable")[:probes]
            mask = np.isin(self._assignments, probe_order)
            candidate_ids = np.flatnonzero(mask)
            if candidate_ids.size == 0:  # degenerate: fall back to exact
                candidate_ids = np.arange(len(self._vectors))
            scores = np.full(len(self._vectors), -np.inf, dtype=np.float32)
            scores[candidate_ids] = self._matrix()[candidate_ids] @ q
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-np.inf)
        valid = np.flatnonzero(np.isfinite(scores))
        order = valid[np.lexsort((valid, -scores[valid]))][:k]
        return [(int(i), float(scores[i])) for i in order]

    def batch_add(self, items: Iterable[tuple[int, np.ndarray]]) -> None:
        for _, vec in items:
            self.add([vec])
