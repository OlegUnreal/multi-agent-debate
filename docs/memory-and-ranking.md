# Memory, retrieval and ranking — derivations

The numbers and formulas behind the table in the README. Everything here is
implemented in `debate/embeddings.py`, `vector_index.py`, `memory.py`,
`episodic.py`, `ranking.py` and `rubric.py`, and pinned by
`tests/test_ranking.py`, `tests/test_embeddings_index.py`,
`tests/test_retrieval_memory.py`, `tests/test_episodic.py` and
`tests/test_rubric.py`.

## 1. Embeddings (`embeddings.py`)

Two regimes, chosen automatically:

- **TF-IDF + SVD.** `TfidfVectorizer(word 1-2 grams + char_wb 2-4 grams)` →
  `TruncatedSVD` (seeded) → L2-normalised float32 rows. Fit per corpus, so the
  projection reflects the vocabulary actually used in the debate.
- **Signed feature hashing.** When the corpus is too small (`< min_docs_for_svd`)
  or SVD variance is degenerate, each token is hashed with `blake2b` into a
  signed `±1` contribution over a fixed dimension: deterministic, offline, and
  no fit state to serialise.

Both paths return unit vectors, so cosine similarity is a plain dot product
downstream.

## 2. Vector index (`vector_index.py`)

- Below `IVF_MIN_DOCS = 128` vectors: exact flat cosine scan. At debate scale
  (hundreds of turns) this is faster than any ANN structure and exactly
  reproducible.
- At ≥ 128 vectors the index self-promotes to IVF: seeded `KMeans` picks
  `nlist ≈ sqrt(n)` centroids, queries probe the `nprobe` nearest centroids
  first. Ties are broken by id, so runs never disagree.

## 3. Agent memory recall (`memory.py`)

For each candidate turn `t` with embedding `e_t`, query embedding `q`, and age
`a_t` (in turns):

```
score(t) = cos(e_t, q) + 0.25 * 0.5^(a_t / half_life)
```

Then greedy **MMR** over the ranked list, penalising redundancy inside the
selection:

```
mmr(t) = score(t) - λ * max_{s in S} sim(e_t, e_s)
λ scaled by role: same-role redundancy ×1.0, cross-role ×0.35
```

Cross-role arguments are less redundant by definition (proposer and critic
rarely repeat each other), so their similarity costs less. The selection is
finally packed into the token budget; if `as_prompt()` is called without a
query the store degrades to the original blind-FIFO behaviour.

## 4. Episodic memory (`episodic.py`)

After a debate, each argument is stored in SQLite with its embedding and its
Bradley–Terry strength (section 5). Before a new debate on a similar topic,
`recall()` scores prior records:

```
score(r) = 1.0 * cos(q, r.embed) + 0.3 * minmax(r.strength)
```

with a raw-cosine `min_similarity` gate so irrelevant history is never injected
as "prior evidence". The formatted block is handed to both agents as context:
strong arguments from past debates literally resurface.

## 5. Bradley–Terry ranking (`ranking.py`)

The judge emits a rubric per round including a `winner` (proposer|critic).
Each round is one pairwise preference. The BT model assigns latent strength
`r_i` to each argument with

```
P(i beats j) = σ(r_i - r_j)
```

fitted by minimising the ridge-regularised negative log-likelihood

```
L(r) = Σ_prefs log(1 + exp(r_loser - r_winner)) + 1e-3 * ||r - mean(r)||²
```

- The ridge term is applied **after centring**, so strengths are identified
  up to translation (BT is shift-invariant) and the optimum is unique.
- Gradient is computed analytically with `scipy.special.expit`; optimisation is
  `L-BFGS-B` with the log-likelihood and gradient-norm history retained.
- Win probabilities come from the fitted `σ(r_i - r_j)`; the ordering is the
  ranking.
- **Anytime fallbacks:** an online Elo/TrueSkill-style `mu/σ` leaderboard
  (`K=32`, updates as rounds arrive) and, before any preferences exist, the
  rubric-total ordering.
- `evaluate_ranking()` scores a fitted ranking against known latent scores via
  Kendall τ and Spearman ρ — the test suite recovers planted orderings from
  noisy preferences.

## 6. Rubric judging and calibration (`rubric.py`)

- The judge must return a Pydantic-validated rubric: five 0–3 ordinal scores
  (evidence, reasoning, rebuttal, clarity) + winner `Literal` + confidence in
  [0,1]. Malformed output goes through a free repair (fence stripping, type
  coercion, clamping) and then one prompted repair with the validation error
  echoed; a regex verdict extraction is the last resort.
- **Expected Calibration Error (ECE)** — judge `confidence` vs whether the BT
  consensus agreed with the judge's winner:

```
ECE = Σ_bins (n_b / N) * |acc(b) - conf(b)|
```

  10 equal-width bins, last bin closed at 1.0. Round-to-round winner stability
  measures how often the judge flip-flops — a sanity signal for rubric noise.

## 7. The experimental `agentic/` package

`agentic/` (browser guard/vision/loop) came in from an earlier experiment and
is **not imported by `debate/`**. Rather than wire it in or delete it,
`tests/test_agentic.py` pins its current semantics (behavior-lock tests): any
change to the guard's URL policy or the vision parser is now a deliberate act.
