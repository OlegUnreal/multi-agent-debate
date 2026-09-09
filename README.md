# multi-agent-debate

Several agents argue about one decision until they reach consensus — now with a web UI.

## Roles

- **Proposer** — argues FOR the decision, has its own memory.
- **Critic** — argues AGAINST, independent context.
- **Judge** — reads both sides, scores, decides CONTINUE / ACCEPT / REJECT.

Each agent has isolated conversation memory, so they never bleed into each other. The judge is the only one allowed to terminate the loop.

## Why this is interesting

Most agent demos are a single LLM with tools. This one shows *orchestration*: independent contexts, adversarial pressure, and a termination condition that isn't just "max steps".

## Architecture

```
config.py        -> Settings from env / .env
logging_config.py-> structured JSON logs (no secrets, extras scrubbed)
agents.py        -> Proposer / Critic / Judge + system prompts
memory.py        -> per-agent rolling memory
llm.py           -> OpenAI client with retries + timeout
loop.py          -> debate loop, verdict extraction, safe speak
server.py        -> FastAPI + static UI, background jobs
static/          -> single-page UI (vanilla JS, no build step)
__main__.py      -> CLI: python -m debate "<topic>"
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env   # then put your OPENAI_API_KEY in .env

# offline demo (no key needed)
python -m debate.demo

# CLI live debate
python -m debate "Should we migrate the monolith to microservices?"

# web UI  ->  http://127.0.0.1:8000
python -m debate.server

# tests
pytest -q
```

## Config

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `DEBATE_MODEL` | `gpt-4o-mini` | model name |
| `DEBATE_MAX_ROUNDS` | `4` | safety cap on rounds |
| `DEBATE_TEMPERATURE` | `0.4` | sampling temperature |
| `DEBATE_TIMEOUT` | `30` | per-call timeout (seconds) |
