"""Per-agent memory: isolated conversation history.

Two read paths share one store:

* ``as_prompt()`` without a query keeps the original blind-FIFO behaviour
  (last ``max_messages`` turns) — the explicit fallback path.
* ``recall(query, k)`` / ``as_prompt(query=...)`` retrieve the most *relevant*
  past turns instead of the most recent ones, blending semantic similarity,
  exponential recency decay and MMR role diversity, then packing the result
  into a token budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .embeddings import TextEmbedder


@dataclass
class Message:
    role: str
    content: str


@dataclass
class _RecallItem:
    seq: int
    message: Message
    semantic: float
    recency: float

    @property
    def score(self) -> float:
        return self.semantic + 0.25 * self.recency


def estimate_tokens(text: str) -> int:
    """Cheap token proxy: ~1.33 tokens per whitespace-separated word."""
    words = len(text.split())
    return max(1, math.ceil(words * 4 / 3)) if words else 0


class AgentMemory:
    """Rolling window + retrieval store for one agent.

    ``max_messages`` bounds the *fallback* FIFO view exactly as before;
    ``max_stored`` bounds the retrieval archive (embeddings kept in-process).
    """

    def __init__(
        self,
        max_messages: int = 40,
        max_stored: int = 500,
        use_retrieval: bool = True,
        token_budget: int = 900,
    ) -> None:
        self._messages: list[Message] = []          # FIFO view (legacy contract)
        self._archive: list[Message] = []           # everything, for recall
        self._archive_start: int = 0                # seq of archive[0]
        self._dirty = True
        self.max_messages = max_messages
        self.max_stored = max_stored
        self.use_retrieval = use_retrieval
        self.token_budget = token_budget

    # -- write path ------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role, content))
        self._archive.append(Message(role, content))
        # Keep the tail so long debates don't blow the context window.
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
        if len(self._archive) > self.max_stored:
            drop = len(self._archive) - self.max_stored
            self._archive = self._archive[drop:]
            self._archive_start += drop
        self._dirty = True

    def clear(self) -> None:
        self._messages.clear()
        self._archive.clear()
        self._archive_start = 0
        self._dirty = True

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def size(self) -> int:
        """Number of turns kept in the retrieval archive (>= len(self))."""
        return len(self._archive)

    # -- legacy fallback path ---------------------------------------------

    def as_prompt(self, query: str | None = None,
                  token_budget: int | None = None) -> list[dict]:
        """Conversation dicts. Without ``query`` this is the blind FIFO."""
        if query is None or not self.use_retrieval or not self._archive:
            return [{"role": m.role, "content": m.content} for m in self._messages]
        try:
            recalled = self.recall(query, k=8, token_budget=token_budget or self.token_budget)
        except Exception:  # noqa: BLE001 — retrieval must never kill a debate
            return [{"role": m.role, "content": m.content} for m in self._messages]
        if not recalled:
            return [{"role": m.role, "content": m.content} for m in self._messages]
        recalled = [m for m in recalled if m.content != query]
        if not recalled:
            return [{"role": m.role, "content": m.content} for m in self._messages[-6:]]
        lines = ["Relevant points from your earlier turns:"]
        lines += [f"- [{m.role}] {m.content}" for m in recalled]
        prompt = [{"role": "user", "content": "\n".join(lines)}]
        prompt += [{"role": m.role, "content": m.content} for m in self._messages[-6:]]
        return prompt

    # -- retrieval path ------------------------------------------------------

    def _recency(self, seq: int, half_life: float) -> float:
        age = (self._archive_start + len(self._archive) - 1) - seq
        return 0.5 ** (max(age, 0) / half_life) if half_life > 0 else 1.0

    def recall(self, query: str, k: int = 5, token_budget: int = 600,
               recency_half_life: float = 24.0, lambda_mult: float = 0.85
               ) -> list[Message]:
        """Top-k archived turns for ``query`` inside ``token_budget`` tokens.

        Ranking = semantic cosine + recency bonus; selection is greedy MMR
        over the blended score, where redundancy is penalised by embedding
        similarity *and* duplicated role, so the final set mixes turns.
        Returns messages in chronological order.
        """
        if not self._archive or k <= 0:
            return []
        texts = [m.content for m in self._archive]
        emb = TextEmbedder().fit(texts + [query])
        vecs = emb.transform(texts + [query])
        doc_vecs, q_vec = vecs[:-1], vecs[-1]
        sem = doc_vecs @ q_vec
        items = [
            _RecallItem(seq=i, message=m, semantic=float(s),
                        recency=self._recency(i, recency_half_life))
            for i, (m, s) in enumerate(zip(self._archive, sem))
        ]
        items.sort(key=lambda it: (-it.score, it.seq))

        chosen: list[_RecallItem] = []
        remaining = list(items)
        while remaining and len(chosen) < k:
            best, best_val = None, -math.inf
            for cand in remaining:
                redundancy = 0.0
                for sel in chosen:
                    sim = float(doc_vecs[cand.seq] @ doc_vecs[sel.seq])
                    # Same-role near-duplicates hurt diversity most; an
                    # opponent restating a point is only partly redundant.
                    factor = 1.0 if cand.message.role == sel.message.role else 0.35
                    redundancy = max(redundancy, sim * factor)
                val = lambda_mult * cand.score - (1 - lambda_mult) * redundancy
                if val > best_val + 1e-12:
                    best, best_val = cand, val
            chosen.append(best)
            remaining.remove(best)

        # Pack into the token budget; drop the least valuable chosen items.
        chosen.sort(key=lambda it: (-it.score, it.seq))
        kept: list[_RecallItem] = []
        budget = token_budget
        for it in chosen:
            cost = estimate_tokens(it.message.content)
            if kept and cost > budget:
                continue
            if not kept and cost > token_budget:
                continue  # even alone it does not fit — skip, try smaller
            kept.append(it)
            budget -= cost
        kept.sort(key=lambda it: it.seq)
        return [it.message for it in kept]
