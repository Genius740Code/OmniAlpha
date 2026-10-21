"""Sequential Probability Ratio Test (SPRT) for gating two engines (serve-side Elo pick).

Standard Wald SPRT on the logistic Elo model:
  H0: elo_A - elo_B == elo0   (e.g. candidate is NOT better)
  H1: elo_A - elo_B == elo1   (e.g. candidate is better by elo1), with elo1 > elo0
Draws are split evenly into 0.5 wins + 0.5 losses (Fishtest convention), which keeps the
Bernoulli log-likelihood-ratio form. Dependency-free (stdlib only) so it is unit-testable.

Used by the orchestrator to decide champion promotion only on a confirmed gain, instead
of blindly copying latest -> champion. A `continue` verdict means: run more games.
"""
import math


def elo_to_score(elo):
    """Expected score (win + 0.5*draw) for a player `elo` points stronger, logistic model."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt(wins, losses, draws=0, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05):
    """Return (decision, llr) for the observed tally.

    decision is one of:
      'accept_h1'  candidate is stronger than champion by >= elo1  (promote)
      'accept_h0'  candidate is not stronger by elo1               (keep champion)
      'continue'   insufficient evidence; run more games
    llr is the log-likelihood ratio log(P(data|H1) / P(data|H0)).
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
        raise ValueError("alpha/beta must be in (0,1)")
    if elo1 <= elo0:
        raise ValueError("elo1 must exceed elo0")
    if wins < 0 or losses < 0 or draws < 0:
        raise ValueError("counts must be non-negative")
    p0 = elo_to_score(elo0)
    p1 = elo_to_score(elo1)
    if not (0.0 < p0 < 1.0) or not (0.0 < p1 < 1.0):
        raise ValueError("elo0/elo1 map to degenerate win probabilities")
    w = wins + 0.5 * draws
    l = losses + 0.5 * draws
    llr = w * math.log(p1 / p0) + l * math.log((1.0 - p1) / (1.0 - p0))
    hi = math.log((1.0 - beta) / alpha)   # accept H1 (promote) at/above this
    lo = math.log(beta / (1.0 - alpha))   # accept H0 (keep)     at/below this
    if llr >= hi:
        return "accept_h1", llr
    if llr <= lo:
        return "accept_h0", llr
    return "continue", llr


def promote_decision(decision, stats):
    """Map an SPRT verdict to a champion-promotion boolean.

    accept_h1  -> promote (candidate confirmed stronger)
    accept_h0  -> keep incumbent
    max_games  -> undecided within the game cap: promote only if the candidate has the
                  higher raw win count (tie keeps the incumbent)
    """
    if decision == "accept_h1":
        return True
    if decision == "accept_h0":
        return False
    return stats.get("w", 0) > stats.get("l", 0)


def bradley_terry_elo(names, pair_results, max_iter=500, tol=1e-10, eps=1e-6):
    """Fit Elo ratings from a full pairwise table with a Bradley-Terry MM iteration.

    Used by the orchestrator's ordered endgame (`eval.ordered`): round-robin the final
    checkpoints, then rank them by maximum-likelihood strength instead of raw score.

    names: candidate names, index -> name. pair_results: {(i, j): (w_i, l_i, draws)}
    from i's point of view with i < j. Draws count as half a win for each side
    (same convention as sprt()). Returns {name: elo} with the played field centred
    at 0 Elo (geometric mean strength 1); players with no games get 0.0.

    Iteration is Hunter's MM update (Zermelo's algorithm with half-win draws):
        gamma_i <- W_i / sum_{j != i} n_ij / (gamma_i + gamma_j)
    which converges monotonically to the MLE. `eps` clamps gamma away from 0 so a
    winless player stays finite (-inf would poison the whole table).
    """
    k = len(names)
    n_ij = [[0.0] * k for _ in range(k)]
    w_ij = [[0.0] * k for _ in range(k)]
    for (i, j), (w, l, d) in pair_results.items():
        n = w + l + d
        if n <= 0:
            continue
        n_ij[i][j] += n; n_ij[j][i] += n
        w_ij[i][j] += w + 0.5 * d
        w_ij[j][i] += l + 0.5 * d
    played = [sum(n_ij[i]) > 0.0 for i in range(k)]
    gamma = [1.0] * k
    for _ in range(max_iter):
        new = list(gamma)
        change = 0.0
        for i in range(k):
            if not played[i]:
                continue
            denom = sum(n_ij[i][j] / (gamma[i] + gamma[j])
                        for j in range(k) if j != i and n_ij[i][j] > 0.0)
            if denom <= 0.0:
                continue
            new[i] = max(eps, sum(w_ij[i]) / denom)
            change = max(change, abs(new[i] - gamma[i]) / max(gamma[i], eps))
        gamma = new
        if change < tol:
            break
    vals = [g for g, p in zip(gamma, played) if p]
    if vals:
        geo = math.exp(sum(math.log(max(g, eps)) for g in vals) / len(vals))
        if geo > 0.0:
            gamma = [g / geo for g in gamma]
    return {names[i]: (400.0 * math.log10(max(gamma[i], eps)) if played[i] else 0.0)
            for i in range(k)}
