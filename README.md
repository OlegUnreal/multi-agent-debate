# multi-agent-debate

Several agents argue about one decision until they reach consensus — with a live web UI to watch them.

## The idea

One LLM with tools is a demo. Three LLMs with *independent memory*, adversarial roles, and a termination condition that isn't just "max steps" is an orchestration problem. This repo is that problem, solved.

- **Proposer** argues FOR the decision, keeps its own rolling memory.
- **Critic** argues AGAINST, with a completely separate context — it never sees the proposer's private chain-of-thought.
- **Judge** reads both sides, scores them, and is the *only* agent allowed to say CONTINUE / ACCEPT / REJECT.
- **Memory retrieves, it does not just truncate** — each agent recalls the most *relevant* past turns (semantic similarity + recency + role diversity, packed into a token budget), and strong arguments survive across debates in SQLite.
- **Arguments are ranked, not just judged** — the judge's pairwise preferences are fit with Bradley–Terry MLE, so "which argument actually won this debate" is a calibrated number.

The loop ends when the judge accepts, rejects, or the safety cap on rounds is hit.

## Why this is interesting

- **Isolated contexts** — agents cannot bleed into each other; memory is per-role, not shared.
- **Adversarial pressure** — the critic exists to find holes, which surfaces weak arguments the proposer would otherwise gloss over.
- **Real termination** — the judge decides, not a counter. This mirrors how human review panels actually work.
- **Observable** — a single-page web UI streams every round in real time.

## Architecture

```
debate/
├── config.py           # Settings from env / .env — stdlib frozen dataclass + dotenv
├── logging_config.py   # structured JSON logs, secrets scrubbed, extras allow-listed
├── agents.py           # Proposer / Critic / Judge + role-specific system prompts
├── embeddings.py       # deterministic offline text embeddings (TF-IDF+SVD, hash fallback)
├── vector_index.py     # in-process ANN: flat cosine, self-promotes to IVF past 128 docs
├── memory.py           # per-agent store: retrieval recall + legacy blind-FIFO fallback
├── episodic.py         # cross-debate SQLite memory ("prior evidence" recall)
├── ranking.py          # Bradley–Terry MLE + online Elo/TrueSkill-style leaderboard
├── rubric.py           # Pydantic judge rubric, malformed-output repair, ECE, consistency
├── llm.py              # OpenAI client: retries, timeout, injectable for tests
├── loop.py             # debate loop, verdict extraction, ranking + calibration hooks
├── server.py           # FastAPI + static UI, threaded background jobs, polling
├── static/index.html   # vanilla JS UI, no build step
├── __main__.py         # CLI: python -m debate "<topic>"
├── demo.py             # offline stub (no key)
└── demo_llm.py         # live run with real OpenAI

agentic/                # EXPERIMENTAL, not imported by debate/ — see docs/memory-and-ranking.md
```

### Data flow

```
Topic ──► Proposer.speak() ──► Critic.speak() ──► Judge.speak()
              ▲    ▲               ▲    ▲               │
              │    └── memory.recall(query, k) ─────────┤  relevant turns,
              └──────────── per-role isolated store ────┘  not a blind tail
                                                         │
                          verdict: CONTINUE | ACCEPT | REJECT
                          rubric: {evidence, reasoning, rebuttal, clarity,
                                   winner, confidence}      [rubric_judging=True]
                                             │
        after the loop ──────────────────────▼
   preferences (winner,loser per round) ──► Bradley–Terry MLE ──► argument strengths
   judge confidences ─────────────────────► ECE / winner stability ──► /metrics
   arguments + strengths ─────────────────► SQLite episodic store ──► prior evidence
                                            for the NEXT debate
```

## How to run

```bash
git clone https://github.com/OlegUnreal/multi-agent-debate.git
cd multi-agent-debate

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # put OPENAI_API_KEY=sk-... in .env

# offline demo (no key)
python -m debate.demo

# CLI live debate
python -m debate "Should we migrate the monolith to microservices?"

# web UI  ->  http://127.0.0.1:8000
python -m debate.server

# tests
pytest -q
```

Windows notes:

- Activate with `.venv\Scripts\activate`.
- The web UI binds to `127.0.0.1:8000` by default; change `DEBATE_HOST` / `DEBATE_PORT` if the port is taken.
- No native binaries required — pure Python + the OpenAI HTTP client.

## Libraries used and why

| Library | Version | Why it is here |
|---|---|---|
| `openai` | `>=1.40` | Typed client for the three agent roles. One shared client instance, role-specific system prompts passed per call — no per-role connection overhead. |
| `fastapi` | `>=0.110` | ASGI web framework + request/response models. Debate handlers are synchronous, so each job runs in a worker thread and the polling endpoint stays responsive. |
| `pydantic` | `>=2.7` | Judge rubric schema with strict ordinal validation (`ge`/`le`, `Literal` winner), plus FastAPI request/response models. |
| `uvicorn[standard]` | `>=0.27` | ASGI server that runs FastAPI. The `[standard]` extra pulls in `uvloop`/`httptools` for better throughput; fine for a local demo too. |
| `httpx` | `>=0.27` | Async HTTP client used by the test suite to hit the FastAPI app without spinning up a real server. Also the transport FastAPI's `TestClient` is built on. |
| `numpy` | `>=1.26` | Vector maths for embeddings, the cosine index, the Bradley–Terry log-likelihood/gradient and calibration metrics. |
| `scipy` | `>=1.11` | `L-BFGS-B` for the BT fit, `expit` for numerically stable gradients, `kendalltau`/`spearmanr` for ranking evaluation. |
| `scikit-learn` | `>=1.4` | `TfidfVectorizer` + `TruncatedSVD` for embeddings, `KMeans` for the IVF index. Deterministic seeds only — no model downloads. |
| `python-dotenv` | `>=1.0` | Loads `.env` so the API key never lands in source control. |
| `pytest` | `>=8.0` | (dev) Test runner for loop logic, retrieval, ranking recovery, calibration, and server endpoints. |

Why these and not a vector database: the whole retrieval layer is
`TfidfVectorizer` + `TruncatedSVD` + an in-process cosine index. A debate is
hundreds of turns, not hundreds of millions — an embedded index removes the
network dependency, keeps the suite deterministic and offline, and makes the
embedding math reviewable in one file. `sentence-transformers`/`faiss` are
deliberately absent: no weights to download, no GPU, no silent version drift.

Why FastAPI over Flask: the UI polls a job that is still running. FastAPI's
sync route handlers are dispatched to Starlette's threadpool, so polling never
blocks, and job state lives in a dict guarded by one `threading.Lock` —
no connection-state bugs, no task queue. The debate is short enough that
polling latency is invisible; WebSockets are a one-line swap later.

## Web UI

`python -m debate.server` starts FastAPI on port 8000. Open the page, type a topic, hit Start. You see each round as it happens:

- Proposer bubble (green) — the argument FOR.
- Critic bubble (red) — the argument AGAINST.
- Judge bubble (blue) — the judge's raw reply: the rubric JSON plus the verdict line.
- Final banner — ACCEPT / REJECT.

The rubric is returned as structured JSON on `GET /api/debate/{id}` (`rounds[].rubric`, `metrics`), but the page does not render those fields as widgets yet — that is a known gap, not a missing backend.

No build step, no framework — one HTML file with vanilla JS. Designed to be readable in an interview, not to win a design award.

## Memory, retrieval and ranking

Everything below is learned from the data that flows through the debate, with a
heuristic underneath each learned component so the system degrades instead of
failing. Full derivation: [docs/memory-and-ranking.md](docs/memory-and-ranking.md).

| Layer | Learned | Heuristic / fallback |
|---|---|---|
| Embeddings | TF-IDF (word 1-2 + char_wb 2-4 grams) → `TruncatedSVD`, L2-normalised float32, fit per corpus | signed `blake2b` feature hashing when the corpus is too small or zero-variance for SVD to mean anything |
| Vector index | IVF centroids (`KMeans`, seeded) once ≥ 128 vectors | exact flat cosine scan below that; ties broken by id so runs never disagree |
| Agent memory recall | blend of embedding cosine against the current query | `+ 0.25 * 0.5^(age/half_life)` recency term, greedy MMR with same-role redundancy ×1.0 / cross-role ×0.35, then a token-budget pack. `as_prompt()` with no query is still the original blind FIFO |
| Episodic memory | stored embeddings + judged BT strength per argument | min-max-normalised blend `1.0*sim + 0.3*strength`, raw-cosine `min_similarity` gate, SQLite `:memory:` or file |
| Argument ranking | Bradley–Terry MLE (ridge 1e-3, centred, `L-BFGS-B`) with log-likelihood / gradient-norm history | online Elo/TrueSkill-style `mu`/`sigma` leaderboard for anytime use; rubric-total ordering when there are no pairwise preferences yet |
| Judging | Pydantic rubric validated strictly (0-3 ordinals, `Literal` winner, confidence in [0,1]) | free repair (fence stripping, type coercion, clamping) then one prompted repair with the validation error echoed; regex verdict extraction as the last resort |
| Calibration | ECE of judge confidence against the BT consensus winner + round-to-round winner stability | equal-width 10-bin ECE with the last bin closed at 1.0; deterministic given the history |

No torch, no transformers, no sentence-transformers, no faiss, no langgraph —
numpy/scipy/scikit-learn only, so the suite runs offline in about four seconds.

## Measured results

These are the numbers the tests actually assert on the fixtures in `tests/`,
not aspirations. Run `pytest -q` to reproduce.

| Metric | Fixture | Value | Test |
|---|---|---|---|
| BT rank recovery vs injected latent ordering | 5 players, 400 sampled pairings (319 usable) | Kendall τ = **1.0**, Spearman ρ = **1.0**, top-1 = **1.0** | `test_bt_beats_wins_count_baseline_on_rank_correlation` |
| Raw-wins baseline on the same data | " | τ = 0.8, ρ = 0.9, top-1 = 1.0 | same |
| BT ordering + win-matrix symmetry | " | exact order `2 > 0 > 3 > 1 > 4`, W + Wᵀ = 1 | `test_bt_recovers_known_latent_ranking` |
| BT convergence | " | 8 L-BFGS-B iterations, final ‖grad‖ = **7.8e-07**, `converged=True` | `test_convergence_diagnostics_present` |
| Calibrated win probability | r₂ − r₄ = 1.919 | P(2 beats 4) = **0.872** (asserted > 0.85) | `test_bt_recovers_known_latent_ranking` |
| BT vs latent correlation of mean win probability | " | Pearson r > **0.98** | `test_bt_probabilities_are_calibrated_to_latent` |
| Retrieval precision@4 | 12-turn archive, 4 relevant | **1.0** | `test_recall_precision_at_4` |
| Retrieval recall@6 | " | **1.0** (all 4 relevant returned; precision capped at 4/6) | `test_recall_at_6_contains_every_relevant_turn` |
| Token budget respected | k=8, budget 40 | returned turns ≤ 40 tokens | `test_recall_respects_token_budget` |
| Early-argument recovery | window of 6, 12 later turns | turn 0 retrieved from outside the FIFO window | `test_retrieval_recovers_early_argument_a_fifo_would_drop` |
| ECE, hand-checkable case | conf 0.9/0.9/0.6/0.6, labels 1/1/0/0 | **0.35** = 0.5·\|1−0.9\| + 0.5·\|0−0.6\| | `test_ece_hand_checkable_case` |
| Rubric repair | fenced / clamped / non-JSON input | salvaged or rejected, never silently accepted | `test_free_repair_salvages_fenced_and_partial_json` |
| API job flow | `stub_llm`, rubric on | 1 round, verdict + metrics dict, `agreement=1.0` | `test_job_flow_with_rubric_metrics` |

The τ/ρ = 1.0 result is on a *synthetic* fixture with a clear latent ordering
and 400 matches; it says the estimator is correct, not that real judge output
is this clean. The baseline gap (τ 0.8 → 1.0) is the point: raw win counts
lose ordering when the schedule is unbalanced, BT does not.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `DEBATE_MODEL` | `gpt-4o-mini` | model for all three roles |
| `DEBATE_MAX_ROUNDS` | `4` | safety cap on debate rounds |
| `DEBATE_TEMPERATURE` | `0.4` | sampling temperature |
| `DEBATE_TIMEOUT` | `30` | per-call timeout (seconds) |
| `DEBATE_MAX_MEMORY` | `40` | legacy FIFO window per agent (turns replayed when there is no query) |
| `DEBATE_MEMORY_TOKEN_BUDGET` | `900` | budget for the retrieved-turns block injected into a prompt |
| `DEBATE_EPISODIC_DB` | `:memory:` | SQLite path for cross-debate memory; a file path persists across restarts |
| `DEBATE_HOST` / `DEBATE_PORT` | `127.0.0.1` / `8000` | server bind (read by `python -m debate.server`) |

Retrieval and ranking are tuned by keyword arguments, not env vars — they are
algorithm choices, not deployment config:

| Knob | Where | Default | Effect |
|---|---|---|---|
| `k`, `token_budget` | `AgentMemory.recall` | `5`, `600` | how many turns, how much context |
| `recency_half_life` | `AgentMemory.recall` | `24.0` | exponential recency decay, in turns |
| `lambda_mult` | `AgentMemory.recall` | `0.85` | MMR relevance/diversity trade-off |
| `min_similarity` | `EpisodicMemory.recall` | `0.05` | raw-cosine gate on off-topic history |
| `l2`, `max_iter` | `fit_bradley_terry` | `1e-3`, `500` | ridge on ratings, optimiser cap |
| `ivf_threshold` | `VectorIndex` | `128` | exact scan → IVF promotion |

## Testing

```bash
pytest -q
pytest -v tests/test_loop.py      # debate loop + verdict extraction
pytest -v tests/test_server.py    # FastAPI endpoints + UI serving
```

Covered: role isolation, memory truncation, verdict parsing from free text, retry on empty responses, log scrubbing, server job lifecycle.

## Design decisions (interview notes)

1. **Why separate memory per agent?**
   Shared context would let the critic see the proposer's reasoning and just mirror it. Isolation forces genuine adversarial pressure — the whole point of the pattern.

2. **Why is the judge the only one who can terminate?**
   If any agent could stop the loop, the proposer would declare victory on round one. A single authoritative judge mirrors a human review panel.

3. **Why extract the verdict from free text instead of forcing JSON?**
   Models drift from strict schemas under pressure. Parsing `ACCEPT`/`REJECT`/`CONTINUE` with a regex fallback is more robust than `response_format=json_object`, which silently fails on some providers.

4. **Why a background job + polling UI instead of WebSockets?**
   Simpler to reason about, no connection-state bugs, and the debate is short enough that polling latency is invisible. WebSockets are a one-line swap later.

## Project status

Working prototype with real LLM integration, structured logging, web UI, and tests. Not production — no persistence, no auth, no multi-tenancy. Strong portfolio piece for agent-orchestration interviews.

## License

MIT.
