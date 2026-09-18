"""Argument ranking: Bradley–Terry (batch MLE) and Elo (online).

The judge emits pairwise preferences ("proposer beat critic this round");
``fit_bradley_terry`` turns the whole batch into calibrated ratings via
regularised MLE with convergence diagnostics, while ``EloLeaderboard`` gives
an anytime estimate as the debate streams. ``evaluate_ranking`` scores both
against a raw-wins baseline on synthetic ground-truth data.

No torch anywhere: the BT gradient is closed-form and L-BFGS-B (scipy) is
plenty for debate-sized item counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kendalltau, spearmanr

Preference = tuple[int, int]  # (winner index, loser index)


def _pairwise_probs(ratings: np.ndarray) -> np.ndarray:
    diff = ratings[:, None] - ratings[None, :]
    return 1.0 / (1.0 + np.exp(-diff))


@dataclass
class BradleyTerryResult:
    ratings: np.ndarray
    loglik_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    l2_strength: float = 0.0

    @property
    def win_matrix(self) -> np.ndarray:
        """W[i, j] = P(i beats j) under the fitted model."""
        return _pairwise_probs(self.ratings)

    def win_prob(self, i: int, j: int) -> float:
        return float(1.0 / (1.0 + math.exp(-(self.ratings[i] - self.ratings[j]))))

    def probabilities(self) -> np.ndarray:
        """Calibrated per-item probability of beating a random opponent."""
        w = self.win_matrix
        n = w.shape[0]
        if n <= 1:
            return np.ones(n, dtype=np.float64)
        mask = ~np.eye(n, dtype=bool)
        return (w * mask).sum(axis=1) / (n - 1)

    def ranking(self) -> list[int]:
        """Item indices, best first; ties broken by index for determinism."""
        return list(np.lexsort((np.arange(len(self.ratings)), -self.ratings)))


def fit_bradley_terry(preferences: list[Preference], n_items: int | None = None,
                      l2: float = 1e-3, max_iter: int = 500) -> BradleyTerryResult:
    """MLE Bradley–Terry fit with ridge regularisation, via L-BFGS-B.

    ``preferences`` is a list of (winner, loser) index pairs. Ratings are
    identifiable only up to a constant, so we centre them at zero. The small
    ridge term keeps disconnected players finite. Deterministic: zero init.
    """
    pairs = np.asarray(preferences, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        n = int(n_items or 0)
        return BradleyTerryResult(ratings=np.zeros(n), converged=True)
    n = int(n_items if n_items is not None else pairs.max() + 1)
    winners, losers = pairs[:, 0], pairs[:, 1]

    history: list[float] = []
    grad_hist: list[float] = []

    def neg_loglik_with_grad(r: np.ndarray):
        diff = r[winners] - r[losers]
        # NLL = -log sigmoid(diff) = log(1 + exp(-diff)), computed stably
        ll = float(np.sum(np.logaddexp(0.0, -diff)))
        penalty = 0.5 * l2 * float(r @ r)
        # d/dr[w] log(1+exp(-diff)) = -sigmoid(-diff); stable via expit
        push = expit(-diff)                        # P(loser upsets winner)
        g = np.zeros(n, dtype=np.float64)
        np.add.at(g, winners, -push)
        np.add.at(g, losers, push)
        g += l2 * r
        g -= g.mean()                               # keep the centring manifold
        history.append(-ll)
        grad_hist.append(float(np.linalg.norm(g)))
        return ll + penalty, g

    res = minimize(neg_loglik_with_grad, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": 1e-8, "ftol": 1e-12,
                            "maxls": 40})
    ratings = np.asarray(res.x, dtype=np.float64)
    ratings -= ratings.mean()
    last_grad = grad_hist[-1] if grad_hist else float("nan")
    return BradleyTerryResult(
        ratings=ratings,
        loglik_history=history[-int(res.nit or 0) - 1:],
        grad_norm_history=grad_hist[-int(res.nit or 0) - 1:],
        n_iter=int(res.nit or 0),
        converged=bool(res.success and last_grad < 1e-4),
        l2_strength=l2,
    )


@dataclass
class EloLeaderboard:
    """Online Elo/TrueSkill-style update: mu/sigma per player.

    Win expectation is Bradley–Terry on the means; updates shrink the
    loser's and winner's confidence intervals (K scales the step, the
    sigma decay mimics TrueSkill's dynamics factor without full message
    passing — deliberate: debate-sized data does not warrant more).
    """
    k: float = 16.0
    beta: float = 400.0
    dynamics: float = 1.005  # sigma inflation per idle period (TrueSkill tau-like)
    start_mu: float = 1000.0
    start_sigma: float = 250.0
    mu: dict[int, float] = field(default_factory=dict)
    sigma: dict[int, float] = field(default_factory=dict)

    def expected_score(self, i: int, j: int) -> float:
        di = self.mu.get(i, self.start_mu) - self.mu.get(j, self.start_mu)
        denom = math.sqrt(2 * self.beta ** 2
                          + self.sigma.get(i, self.start_sigma) ** 2
                          + self.sigma.get(j, self.start_sigma) ** 2)
        return 1.0 / (1.0 + 10 ** (-di / denom))

    def update(self, winner: int, loser: int) -> tuple[float, float]:
        for p in (winner, loser):
            self.mu.setdefault(p, self.start_mu)
            self.sigma.setdefault(p, self.start_sigma)
            self.sigma[p] *= self.dynamics
        exp_w = self.expected_score(winner, loser)
        surprise = 1.0 - exp_w
        self.mu[winner] += self.k * surprise
        self.mu[loser] -= self.k * surprise
        # certainty-of-evidence update: both intervals tighten
        self.sigma[winner] *= 0.95
        self.sigma[loser] *= 0.95
        return self.mu[winner], self.mu[loser]

    def ranking(self) -> list[int]:
        players = sorted(self.mu)
        return sorted(players, key=lambda p: (-self.mu[p], p))

    def probabilities(self) -> np.ndarray:
        """Per-player average win expectation vs the field (calibrated-ish)."""
        players = sorted(self.mu)
        n = len(players)
        out = np.ones(n)
        for a, i in enumerate(players):
            s = sum(self.expected_score(i, j) for j in players if j != i)
            out[a] = s / max(n - 1, 1)
        return out


def wins_count_baseline(preferences: list[Preference], n_items: int) -> np.ndarray:
    """The naive 'count of raw wins' baseline, for honest comparisons."""
    counts = np.zeros(n_items)
    for w, _ in preferences:
        counts[w] += 1.0
    return counts


def evaluate_ranking(preferences: list[Preference], latent_scores: np.ndarray,
                     n_items: int | None = None) -> dict[str, dict[str, float]]:
    """Rank correlations / top-1 accuracy of BT and the wins baseline.

    ``latent_scores[i]`` is the (known) true strength of item i on the
    synthetic fixtures; higher is better. top1 = model's best pick is the
    latent best pick.
    """
    n = int(n_items if n_items is not None else len(latent_scores))
    truth_order = np.argsort(-np.asarray(latent_scores, dtype=float), kind="stable")
    bt = fit_bradley_terry(preferences, n_items=n)
    base = wins_count_baseline(preferences, n)

    def metrics(scores: np.ndarray) -> dict[str, float]:
        tau = kendalltau(scores, latent_scores).statistic
        rho = spearmanr(scores, latent_scores).statistic
        top1 = float(np.argmax(scores) == truth_order[0])
        return {"kendall_tau": float(tau), "spearman_rho": float(rho), "top1": top1}

    return {
        "bradley_terry": metrics(bt.ratings),
        "wins_baseline": metrics(base),
        "bt_converged": float(bt.converged),
    }
