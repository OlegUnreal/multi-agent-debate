"""Tests for the retrieval-augmented AgentMemory.

Covers the instructed headline numbers: precision of recall on a synthetic
fixture with known relevance, token-budget packing, role diversity via MMR,
and the untouched FIFO fallback.
"""
from __future__ import annotations

from debate.memory import AgentMemory, estimate_tokens
from debate.embeddings import TextEmbedder


# Known relevant set: turns about cache eviction / p99 latency (RELEVANT).
# Distractors: OAuth2 tokens, redis migration, CI flakiness.
TURNS = [
    ("user", "Round 1. The cache eviction switch from LRU to random caused the p99 latency regression."),
    ("assistant", "Agreed: random eviction misses the hot cache keys, so p99 latency on cache reads exploded."),
    ("user", "Round 2. OAuth2 refresh tokens must rotate before the access token expiry window closes."),
    ("assistant", "Refresh token rotation guards the session against replay before expiry."),
    ("user", "Round 3. Migrating session storage to redis needs a dual-write cutover with reconciliation."),
    ("assistant", "Dual writes into redis risk divergence, so run a reconciliation job every hour."),
    ("user", "Round 4. Restoring LRU cache eviction is the fastest route to pre-regression p99 latency."),
    ("assistant", "Rolling the eviction policy back to LRU restores latency on the hot cache keys within minutes."),
    ("user", "Round 5. The CI suite flakes on timezone tests unrelated to latency or cache behaviour."),
    ("assistant", "Freeze the clock in the CI timezone test to stop the flakes."),
    ("user", "Round 6. Capacity headroom says the fleet handles twice peak traffic today."),
    ("assistant", "Twice-peak headroom means the rollout needs no extra machines."),
]
RELEVANT = {0, 1, 6, 7}
QUERY = "why did the cache eviction change hurt p99 latency and how do we restore it"


def _fill(m: AgentMemory) -> None:
    for role, content in TURNS:
        m.add(role, content)


def _precision(hits, k):
    idx = {TURNS.index((h.role, h.content)) for h in hits}
    return len(idx & RELEVANT) / k, idx


def test_embeddings_reach_dim_on_this_corpus():
    # guards the fixture: small corpora collapse SVD to a tiny space
    emb = TextEmbedder(dim=32).fit([c for _, c in TURNS] + [QUERY])
    assert emb.backend == "tfidf_svd"


def test_recall_precision_at_4():
    m = AgentMemory()
    _fill(m)
    hits = m.recall(QUERY, k=4, token_budget=10_000)
    precision, idx = _precision(hits, 4)
    assert precision == 1.0, f"precision@4={precision} selected={idx}"


def test_recall_at_6_contains_every_relevant_turn():
    m = AgentMemory()
    _fill(m)
    hits = m.recall(QUERY, k=6, token_budget=10_000)
    precision, idx = _precision(hits, 6)
    # only 4 turns are relevant, so precision@6 is capped at 4/6; the honest
    # claim is recall@6 = 1.0: nothing relevant is missed within the top 6
    assert RELEVANT <= idx, f"missed {RELEVANT - idx}, selected={idx}"
    assert precision == 4 / 6


def test_recall_respects_token_budget():
    m = AgentMemory()
    _fill(m)
    hits = m.recall(QUERY, k=8, token_budget=40)
    total = sum(estimate_tokens(h.content) for h in hits)
    assert 0 < len(hits) < len(TURNS)
    assert total <= 40 or len(hits) == 1


def test_recall_mixes_roles_via_mmr():
    m = AgentMemory()
    _fill(m)
    hits = m.recall(QUERY, k=4, token_budget=10_000)
    assert {h.role for h in hits} == {"user", "assistant"}


def test_as_prompt_with_query_injects_context_without_echo():
    m = AgentMemory()
    _fill(m)
    query = "how do we restore p99 latency after the eviction change"  # not archived yet
    prompt = m.as_prompt(query=query)
    context_block = prompt[0]["content"]
    assert context_block.startswith("Relevant points from your earlier turns:")
    assert "cache eviction" in context_block      # early relevant turn recovered
    assert query not in context_block             # no echo of the live turn
    assert prompt[-1]["content"] == TURNS[-1][1]  # recent tail preserved


def test_fifo_fallback_without_query_is_unchanged():
    m = AgentMemory(max_messages=4)
    for i in range(10):
        m.add("user", f"msg {i}")
    assert len(m) == 4
    prompt = m.as_prompt()
    assert [p["content"] for p in prompt] == [f"msg {i}" for i in range(6, 10)]
    assert m.size == 10  # archive still holds everything for recall


def test_use_retrieval_false_always_falls_back():
    m = AgentMemory(use_retrieval=False)
    _fill(m)
    prompt = m.as_prompt(query=QUERY)
    assert prompt == [{"role": r, "content": c} for r, c in TURNS[-40:]]


def test_recall_is_deterministic():
    a, b = AgentMemory(), AgentMemory()
    _fill(a)
    _fill(b)
    assert [h.content for h in a.recall(QUERY, k=3)] == [h.content for h in b.recall(QUERY, k=3)]


def test_retrieval_recovers_early_argument_a_fifo_would_drop():
    m = AgentMemory(max_messages=6)  # tiny window on purpose
    long_early = "The cache eviction switch from LRU to random caused the p99 latency regression."
    m.add("user", long_early)
    for i in range(12):
        m.add("assistant", f"Filler remark number {i} about totally unrelated rollout logistics.")
    hits = m.recall("what explains the p99 latency regression", k=3, token_budget=10_000)
    assert any(h.content == long_early for h in hits)
    # and the FIFO view genuinely lost it
    assert all(long_early != p["content"] for p in m.as_prompt())
