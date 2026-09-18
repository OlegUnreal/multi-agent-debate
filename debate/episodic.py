"""Cross-debate episodic memory: durable arguments in SQLite.

Each argued turn is stored with its embedding and — once the debate is
ranked — its judged strength (Bradley–Terry rating). Before a new debate on
a *related* topic we recall the strongest historical arguments and inject
them as attributed "prior evidence".

Embeddings are TF-IDF+SVD and therefore corpus-dependent: on recall we
re-fit one embedder over (stored texts + query) so similarity is always
computed in a single shared space. The stored blob is kept for durability
and cheap reuse when the recorded dim happens to match.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .embeddings import TextEmbedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS arguments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id  TEXT    NOT NULL,
    topic      TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    embedding  BLOB,
    dim        INTEGER,
    strength   REAL,
    created_at REAL    NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_arguments_topic ON arguments(topic);
"""


@dataclass(frozen=True)
class EpisodicRecord:
    id: int
    debate_id: str
    topic: str
    role: str
    text: str
    strength: float | None
    similarity: float

    @property
    def attribution(self) -> str:
        return f"prior evidence from debate on '{self.topic}' ({self.role})"


class EpisodicMemory:
    """SQLite-backed store; pass ':memory:' for deterministic tests."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM arguments").fetchone()
        return int(n)

    # -- write path -------------------------------------------------------

    def record(self, debate_id: str, topic: str, role: str, text: str,
               embedding: np.ndarray | None = None) -> int:
        blob = None if embedding is None else np.asarray(embedding, np.float32).tobytes()
        dim = None if embedding is None else int(np.asarray(embedding).size)
        cur = self._conn.execute(
            "INSERT INTO arguments (debate_id, topic, role, text, embedding, dim) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (debate_id, topic, role, text, blob, dim),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def set_strength(self, arg_id: int, strength: float) -> None:
        self._conn.execute("UPDATE arguments SET strength = ? WHERE id = ?",
                           (float(strength), arg_id))
        self._conn.commit()

    # -- read path ----------------------------------------------------------

    def all_records(self) -> list[EpisodicRecord]:
        rows = self._conn.execute(
            "SELECT id, debate_id, topic, role, text, strength FROM arguments ORDER BY id"
        ).fetchall()
        return [EpisodicRecord(r[0], r[1], r[2], r[3], r[4], r[5], 0.0) for r in rows]

    def recall(self, query_topic: str, k: int = 3, min_similarity: float = 0.05,
               exclude_debate: str | None = None,
               similarity_weight: float = 1.0, strength_weight: float = 0.3
               ) -> list[EpisodicRecord]:
        """Strongest historical arguments relevant to ``query_topic``.

        Similarities and strengths are each min-max normalised to [0, 1]
        before blending (``similarity_weight`` / ``strength_weight``) so
        relevance leads and judged strength tie-breaks *within* the set of
        on-topic arguments — off-topic ones cannot win on strength alone.
        Ties break on id for determinism.
        """
        rows = self._conn.execute(
            "SELECT id, debate_id, topic, role, text, strength FROM arguments "
            "ORDER BY id"
        ).fetchall()
        candidates = [r for r in rows if exclude_debate is None or r[1] != exclude_debate]
        if not candidates or k <= 0:
            return []
        texts = [r[4] for r in candidates]
        emb = TextEmbedder().fit(texts + [query_topic])
        vecs = emb.transform(texts + [query_topic])
        sims = vecs[:-1] @ vecs[-1]
        strengths = np.array([r[5] if r[5] is not None else 0.0 for r in candidates])
        sim_n = _minmax(sims)
        if len(candidates) > 1:
            pooled = np.std(strengths)
            z = (strengths - np.mean(strengths)) / (pooled if pooled > 1e-9 else 1.0)
        else:
            z = np.zeros(len(candidates))
        blended = similarity_weight * sim_n + strength_weight * _minmax(z)
        order = np.lexsort((np.array([r[0] for r in candidates]), -blended))
        out: list[EpisodicRecord] = []
        for i in order:
            if float(sims[i]) < min_similarity:
                continue
            r = candidates[int(i)]
            out.append(EpisodicRecord(r[0], r[1], r[2], r[3], r[4], r[5], float(sims[i])))
            if len(out) >= k:
                break
        return out

    def seen_topics(self) -> Sequence[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT topic FROM arguments ORDER BY topic"
        ).fetchall()
        return [r[0] for r in rows]


def format_prior_evidence(records: Sequence[EpisodicRecord], max_chars: int = 900) -> str:
    """Render recalled records as an attributed block for the prompt."""
    if not records:
        return ""
    lines = ["Prior evidence from earlier debates (attributed, weigh accordingly):"]
    used = 0
    for rec in records:
        body = f"- {rec.text.strip()}  [{rec.attribution}, strength={_fmt(rec.strength)}]"
        if used + len(body) > max_chars:
            break
        lines.append(body)
        used += len(body)
    return "\n".join(lines)


def _fmt(strength: float | None) -> str:
    return "n/a" if strength is None else f"{strength:.2f}"


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros(len(a), dtype=np.float64)
    return (a - lo) / (hi - lo)
