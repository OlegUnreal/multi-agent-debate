# multi-agent-debate

Several agents argue about one decision until they reach consensus — with a live web UI to watch them.

## The idea

One LLM with tools is a demo. Three LLMs with *independent memory*, adversarial roles, and a termination condition that isn't just "max steps" is an orchestration problem. This repo is that problem, solved.

- **Proposer** argues FOR the decision, keeps its own rolling memory.
- **Critic** argues AGAINST, with a completely separate context — it never sees the proposer's private chain-of-thought.
- **Judge** reads both sides, scores them, and is the *only* agent allowed to say CONTINUE / ACCEPT / REJECT.

The loop ends when the judge accepts, rejects, or the safety cap on rounds is hit.

## Why this is interesting

- **Isolated contexts** — agents cannot bleed into each other; memory is per-role, not shared.
- **Adversarial pressure** — the critic exists to find holes, which surfaces weak arguments the proposer would otherwise gloss over.
- **Real termination** — the judge decides, not a counter. This mirrors how human review panels actually work.
- **Observable** — a single-page web UI streams every round in real time.

## Architecture

```
debate/
├── config.py          # Settings from env / .env (pydantic-settings)
├── logging_config.py  # structured JSON logs, secrets scrubbed, extras allow-listed
├── agents.py          # Proposer / Critic / Judge + role-specific system prompts
├── memory.py          # per-agent rolling memory (last N turns)
├── llm.py             # OpenAI client: retries, timeout, injectable for tests
├── loop.py            # debate loop, verdict extraction, safe speak()
├── server.py          # FastAPI + static UI, background jobs, SSE-ish streaming
├── static/index.html  # vanilla JS UI, no build step
├── __main__.py        # CLI: python -m debate "<topic>"
├── demo.py            # offline stub (no key)
└── demo_llm.py        # live run with real OpenAI
```

### Data flow

```
Topic ──► Proposer.speak() ──► Critic.speak() ──► Judge.speak()
                ▲                     │                    │
                └──── memory ─────────┘                    │
                                                         ▼
                                              verdict: CONTINUE | ACCEPT | REJECT
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
| `fastapi` | `>=0.110` | ASGI web framework for the debate server. Async-native, so background debate jobs and the polling endpoint share one event loop without threads. |
| `uvicorn[standard]` | `>=0.27` | ASGI server that runs FastAPI. The `[standard]` extra pulls in `uvloop`/`httptools` for better throughput; fine for a local demo too. |
| `httpx` | `>=0.27` | Async HTTP client used by the test suite to hit the FastAPI app without spinning up a real server. Also the transport FastAPI's `TestClient` is built on. |
| `python-dotenv` | `>=1.0` | Loads `.env` so the API key never lands in source control. |
| `pytest` | `>=8.0` | (dev) Test runner for loop logic, verdict parsing, memory truncation, and server endpoints. |

Why FastAPI over Flask: the debate is I/O-bound (waiting on LLM calls), and FastAPI's async model lets the UI poll for updates without blocking the running job. Flask would need threads or a task queue for the same thing.

## Web UI

`python -m debate.server` starts FastAPI on port 8000. Open the page, type a topic, hit Start. You see each round as it happens:

- Proposer bubble (green) — the argument FOR.
- Critic bubble (red) — the argument AGAINST.
- Judge bubble (blue) — the verdict and score.
- Final banner — ACCEPT / REJECT with a one-line summary.

No build step, no framework — one HTML file with vanilla JS. Designed to be readable in an interview, not to win a design award.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `DEBATE_MODEL` | `gpt-4o-mini` | model for all three roles |
| `DEBATE_MAX_ROUNDS` | `4` | safety cap on debate rounds |
| `DEBATE_TEMPERATURE` | `0.4` | sampling temperature |
| `DEBATE_TIMEOUT` | `30` | per-call timeout (seconds) |
| `DEBATE_MEMORY_TURNS` | `6` | rolling memory window per agent |

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
