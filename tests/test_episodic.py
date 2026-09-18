"""Cross-debate episodic memory over SQLite (:memory:, deterministic)."""
from __future__ import annotations

from debate.episodic import EpisodicMemory, format_prior_evidence


ARGS_A = [  # debate A: cache eviction latency
    ("proposer", "Restoring LRU cache eviction recovers p99 latency within minutes."),
    ("critic", "Rolling the eviction policy back re-introduces the memory blowup LRU caused."),
]
ARGS_B = [  # debate B: unrelated, oauth sessions
    ("proposer", "Rotate OAuth2 refresh tokens every third call to cut session bugs."),
    ("critic", "Token rotation every third call doubles auth service load."),
]


def _seed(store: EpisodicMemory) -> None:
    for role, text in ARGS_A:
        store.record("d1", "Should we roll back the cache eviction change?", role, text)
    for role, text in ARGS_B:
        store.record("d2", "Should we rotate OAuth2 refresh tokens aggressively?", role, text)
    store.set_strength(1, 2.0)   # strong argument
    store.set_strength(2, -1.0)
    store.set_strength(3, 0.5)
    store.set_strength(4, -2.0)


def test_record_and_count():
    store = EpisodicMemory(":memory:")
    _seed(store)
    assert len(store) == 4
    assert set(store.seen_topics()) == {
        "Should we roll back the cache eviction change?",
        "Should we rotate OAuth2 refresh tokens aggressively?",
    }


def test_recall_prefers_related_topic_and_strongest_argument():
    store = EpisodicMemory(":memory:")
    _seed(store)
    hits = store.recall("Does rolling back cache eviction fix the latency regression?", k=2)
    assert len(hits) == 2
    topics = {h.topic for h in hits}
    assert topics == {"Should we roll back the cache eviction change?"}
    # strength tie-break: arg 1 (strength 2.0) must outrank arg 2 (-1.0)
    assert hits[0].id == 1
    assert hits[0].text == ARGS_A[0][1]
    assert hits[0].attribution.startswith("prior evidence from debate on")
    assert "cache eviction" in hits[0].attribution


def test_recall_excludes_current_debate_and_is_deterministic():
    store = EpisodicMemory(":memory:")
    _seed(store)
    # query overlaps d2's "rotate refresh tokens" texts; d1 is excluded
    q = "Should we roll back the eviction policy or rotate refresh tokens?"
    r1 = store.recall(q, k=4, exclude_debate="d1")
    assert {r.debate_id for r in r1} == {"d2"}
    r2 = store.recall(q, k=4, exclude_debate="d1")
    assert [r.id for r in r1] == [r.id for r in r2]


def test_strength_changes_the_ordering():
    store = EpisodicMemory(":memory:")
    _seed(store)
    # this query hits both d1 args almost equally hard (sims 0.672 / 0.743),
    # so judged strength - not wording - decides the winner
    q = "does the LRU cache eviction policy restore latency or blow up memory"
    assert store.recall(q, k=2)[0].id == 1  # strength 2.0 beats -1.0
    store.set_strength(2, 10.0)             # make #2 dominate
    hits = store.recall(q, k=2)
    assert hits[0].id == 2


def test_empty_store_and_format_prior_evidence():
    store = EpisodicMemory(":memory:")
    assert store.recall("anything", k=3) == []
    records = [type("R", (), {
        "text": "LRU rollback restores latency",
        "topic": "t", "role": "proposer", "strength": 1.5,
        "attribution": "prior evidence from debate on 't' (proposer)",
    })()]
    block = format_prior_evidence(records)
    assert block.startswith("Prior evidence")
    assert "LRU rollback restores latency" in block
    assert "strength=1.5" in block
    assert format_prior_evidence([]) == ""


def test_embeddings_persisted(tmp_path):
    import numpy as np
    db = tmp_path / "episodic.sqlite3"
    store = EpisodicMemory(db)
    vec = np.arange(8, dtype=np.float32)
    rid = store.record("d9", "topic", "proposer", "text", embedding=vec)
    rows = store._conn.execute(
        "SELECT embedding, dim FROM arguments WHERE id = ?", (rid,)).fetchone()
    assert rows[1] == 8
    np.testing.assert_allclose(np.frombuffer(rows[0], dtype=np.float32), vec)
