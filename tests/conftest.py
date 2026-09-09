import sys
from pathlib import Path

# Make `import debate` work when pytest is run from the repo root
# without an editable install (CI installs only requirements*.txt).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture
def stub_llm():
    """Deterministic LLM: proposer/critic/judge cycle ends in ACCEPT."""
    seq = iter([
        "for it, evidence is strong",
        "against it, hidden costs",
        "ACCEPT, proposer wins on evidence",
    ])
    return lambda _system, _prompt: next(seq)
