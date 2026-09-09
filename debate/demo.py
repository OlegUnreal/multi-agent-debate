"""Demo with a stub LLM so it runs offline."""
from __future__ import annotations

from .loop import run_debate


def stub_llm(system: str, messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    if "Decide" in last:
        return "ACCEPT: proposer evidence outweighs critic concerns."
    if "Counter" in last:
        return "Risk: no rollback plan if the decision fails in production."
    return "Proposal: adopt the change, phased rollout over two weeks."


def main() -> None:
    rounds = run_debate("Should we migrate the auth service to OAuth2?", stub_llm)
    for r in rounds:
        print(f"--- round {r.n} ---")
        print("P:", r.proposer[:80])
        print("C:", r.critic[:80])
        print("J:", r.judge[:80])


if __name__ == "__main__":
    main()
