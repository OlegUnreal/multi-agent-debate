# multi-agent-debate

Several agents argue about one decision until they reach consensus.

## Roles

- **Proposer** — argues FOR the decision, has access to search + memory.
- **Critic** — argues AGAINST, has access to a different toolset (risk analysis).
- **Judge** — reads both sides, scores, decides continue / accept / reject.

Each agent has its own conversation memory, so they don't bleed into each other. The judge is the only one allowed to terminate the loop.

## Why this is interesting

Most agent demos are a single LLM with tools. This one shows *orchestration*: independent contexts, adversarial pressure, and a termination condition that isn't just "max steps".

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m debate.demo
```
