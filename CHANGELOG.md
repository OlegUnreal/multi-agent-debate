# Changelog

## 0.4.0 — 2026-09-18

- Retrieval-based agent memory: TF-IDF+SVD / hashed embeddings per corpus, blended cosine + recency scoring, role-aware greedy MMR, token-budget packing. `as_prompt()` without a query still degrades to blind FIFO.
- In-process ANN vector index: exact flat cosine below 128 vectors, self-promotes to seeded-KMeans IVF above that; id tie-breaking keeps runs reproducible.
- Cross-debate episodic memory: SQLite store of past arguments with embeddings and Bradley–Terry strengths; `recall()` injects similar, strong prior evidence into new debates behind a cosine gate.
- Argument ranking: judge pairwise preferences fitted with Bradley–Terry MLE (ridge 1e-3, centred, L-BFGS-B with optimisation history) plus an online Elo/TrueSkill-style leaderboard and rubric-total fallback; Kendall τ / Spearman ρ evaluation against planted orderings.
- Structured judging: strict Pydantic rubric (0–3 ordinals, Literal winner, confidence), free-repair + prompted-repair pipeline for malformed output, verdict regex as last resort.
- Calibration: ECE of judge confidence vs BT consensus winner, round-to-round winner stability, exposed on `/api/debate/{id}` metrics.
- Server: threaded background jobs with lock-guarded state, rubric + metrics in the API payload; derivation doc `docs/memory-and-ranking.md`.
- Deps: numpy, scipy, scikit-learn, pydantic. Suite: 85 passing (6 new test modules).
