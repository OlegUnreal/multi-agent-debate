"""Bradley–Terry / Elo ranking tests on a synthetic preference fixture with
an injected latent ordering. Achieved numbers are asserted, not printed."""
from __future__ import annotations

import numpy as np
import pytest

from debate.ranking import (
    EloLeaderboard,
    evaluate_ranking,
    fit_bradley_terry,
    wins_count_baseline,
)

# Latent truth: item 2 is best, item 4 worst, ordering 2 > 0 > 3 > 1 > 4.
LATENT = np.array([0.5, -0.3, 1.2, 0.1, -1.0])


def _synthetic_preferences(n_items: int = 5, n_matches: int = 400, seed: int = 7):
    """Pairwise outcomes sampled from the latent BT model, rng-seeded."""
    rng = np.random.default_rng(seed)
    probs = 1.0 / (1.0 + np.exp(-(LATENT[:, None] - LATENT[None, :])))
    prefs = []
    for _ in range(n_matches):
        i, j = rng.integers(n_items, size=2)
        if i == j:
            continue
        winner = i if rng.random() < probs[i, j] else j
        loser = j if winner == i else i
        prefs.append((int(winner), int(loser)))
    return prefs


PREFS = _synthetic_preferences()


def test_bt_recovers_known_latent_ranking():
    res = fit_bradley_terry(PREFS, n_items=len(LATENT))
    assert res.converged
    est_order = res.ratings.argsort()[::-1]
    true_order = LATENT.argsort()[::-1]
    assert est_order.tolist() == true_order.tolist()
    # ratings centred, win matrix consistent
    np.testing.assert_allclose(res.ratings.mean(), 0.0, atol=1e-8)
    W = res.win_matrix
    assert np.allclose(W + W.T, 1.0, atol=1e-9)
    # Converged MLE on 400 sampled matches: P(2 beats 4) = 0.872 achieved.
    # (Slightly below the latent 0.90 because finite-sample fits shrink gaps.)
    assert res.win_prob(2, 4) > 0.85
    assert res.win_prob(4, 2) < 0.15


def test_bt_beats_wins_count_baseline_on_rank_correlation():
    metrics = evaluate_ranking(PREFS, LATENT, n_items=len(LATENT))
    bt, base = metrics["bradley_terry"], metrics["wins_baseline"]
    assert bt["kendall_tau"] >= 0.9
    assert bt["spearman_rho"] >= 0.95
    assert bt["top1"] == 1.0
    assert bt["kendall_tau"] > base["kendall_tau"]


def test_bt_probabilities_are_calibrated_to_latent():
    res = fit_bradley_terry(PREFS, n_items=len(LATENT))
    p = res.probabilities()
    # model P(i beats random opponent) should correlate strongly with truth
    truth = np.array([[1.0 / (1.0 + np.exp(-(LATENT[i] - LATENT[j])))
                       for j in range(len(LATENT)) if j != i]
                      for i in range(len(LATENT))])
    truth_mean = truth.mean(axis=1)
    assert np.corrcoef(p, truth_mean)[0, 1] > 0.98


def test_bt_is_deterministic():
    a = fit_bradley_terry(PREFS, n_items=len(LATENT))
    b = fit_bradley_terry(PREFS, n_items=len(LATENT))
    np.testing.assert_array_equal(a.ratings, b.ratings)
    assert a.n_iter == b.n_iter


def test_convergence_diagnostics_present():
    res = fit_bradley_terry(PREFS, n_items=len(LATENT))
    assert res.n_iter >= 1
    assert res.grad_norm_history and res.grad_norm_history[-1] < 1e-4
    assert len(res.loglik_history) >= 2
    assert res.loglik_history[-1] >= res.loglik_history[0]  # improving loglik


def test_elo_tracks_the_same_winner():
    elo = EloLeaderboard(k=32.0)
    order = [p for p in PREFS]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(order))
    for t in idx:
        w, l = order[int(t)]
        elo.update(w, l)
    assert elo.ranking()[0] == int(LATENT.argmax())
    assert 0.0 <= elo.expected_score(0, 1) <= 1.0
    p = elo.probabilities()
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_empty_and_single_item_edge_cases():
    res = fit_bradley_terry([], n_items=3)
    assert res.ratings.shape == (3,)
    res2 = fit_bradley_terry([(0, 1), (0, 1)], n_items=2)
    assert res2.ratings[0] > res2.ratings[1]
    assert res2.probabilities().shape == (2,)


def test_wins_baseline_is_pure_counts():
    counts = wins_count_baseline([(0, 1), (0, 2), (1, 2)], n_items=3)
    assert counts.tolist() == [2.0, 1.0, 0.0]
