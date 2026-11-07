# MatchSense AI — model assumptions

## 1. Baseline (unchanged)

The forecast starts from the repository's deployed `ScoreModel`
(`artifacts/season_simulator/model.joblib`): two recency-weighted Poisson GLMs
(alpha 0.03, 1460-day half-life) for home and away goals over 32 leakage-safe
pre-match features (Elo, exponentially weighted scoring rates, venue form,
points per game, rest-day difference, promotion flags, matchday phase, …),
mapped to a 21×21 Dixon–Coles score grid (validation chose rho = 0 and a 0%
logistic blend). MatchSense does not refit it. For the current season it copies
the bundled `SeasonState` (end of 2025-26), calls `start_season(2026, teams)`
and applies completed 2026-27 results with `SeasonState.update` in date batches.
A result is only used if its kickoff + 2 h 30 min is before the prediction time.

The existing multinomial-logistic recipe (`FittedModel`, C = 0.1, 1460-day
half-life) is refitted on the bundled training features and shown as a
comparison only: the walk-forward comparison selected a 0% blend weight.

## 2. Lineup layer

For team *t* against opponent *o*:

```
log λ_t = log λ_t^baseline + b1 (D_t − D̄) + b2 (D_o − D̄) + b3 (G_t − Ḡ) + b4 (G_o − Ḡ) + b5 (Ddef_o − D̄def)
```

* `D` (discontinuity) = 1 − Σ_{i∈XI} τ_i / Σ_{top-11} τ_i, where τ_i is the
  player's exponentially weighted start rate over the team's earlier matches
  (half-life 4 matches).
* `G` (absent goal share) = Σ_i g_i (1 − s_i), where g_i is the player's share of
  the team's decayed recent goals and s_i = 1 if he starts.
* `Ddef` (defensive discontinuity) is the same as `D`, but restricted to the
  opponent's goalkeeper and back line and to its top 5 by τ. Training labels
  come from the Football.TXT line-up layout (group 0 = goalkeeper, group 1 =
  back line: one keeper and 3–5 defenders in all 760 team line-ups of 2025-26).
  Live labels come from FPL positions.
* Features are centred at their training means, so a typical amount of rotation
  leaves the baseline unchanged.

Individual attacking value enters through `G`. Individual defensive quality is
not estimated per player: a defender's absence matters through how established
he is (τ) and whether the usual back line is broken (`Ddef`).

### Evidence (walk-forward, 2025-26, matchday folds)

| Model | Val log loss (150) | Test log loss (160) |
|---|---:|---:|
| Baseline | 1.0322 | 1.0498 |
| Continuity + defence (α=20) | 1.0319 | 1.0495 |
| Player-level ridge effects (best on validation) | 1.0283 | 1.0560 |
| Deployed provisional prior | 1.0295 | 1.0517 |

Player-level effects overfit. In round 2 the one-SE choice (player effects)
was worse than the baseline on test (+0.027 log loss, 95% CI +0.000 to +0.053).
In round 3 the one-SE choice (continuity + defence) was indistinguishable from
the baseline (+0.00003, CI −0.0051 to +0.0053). No candidate improved test log
loss with a confidence interval excluding zero.

**The lineup layer is therefore deployed as a provisional Bayesian model.** Its
Gaussian prior has mean [−0.15, +0.10, −0.25, 0, +0.20] and sd [0.15, 0.15,
0.15, 0.10, 0.15], updated with the 2025-26 likelihood (MAP + Laplace
covariance). Posterior:

| Coefficient | Posterior |
|---|---:|
| b1 | −0.150 ± 0.137 |
| b2 | −0.009 ± 0.138 |
| b3 | −0.187 ± 0.116 |
| b4 | −0.008 ± 0.087 |
| b5 (opponent defensive discontinuity) | +0.046 ± 0.126 |

The data pulled b5 well below its prior: in this sample, a broken back line has
a small and uncertain effect. The provisional model's held-out effect is neutral
(+0.0019, CI −0.0057 to +0.0096). Every response carries this label. Full
numbers are in `artifacts/matchsense/REPORT.md`.

**Disclosure.** Three things were introduced after the first evaluation round
showed player effects failing on test (`REPORT_round1.md`): the one-SE rule, the
goal-share feature and the provisional prior. The defensive term was added in a
third round, after rounds 1–2 had been seen (`REPORT_round2.md`). The test
folds are therefore not a pristine holdout for the final choice.

## 3. Selection (starting-XI) model

Pooled logistic regression (C = 3) for P(start | available) with features:
EW start rate, started last match, starts in last 3, EW minutes share, short
rest (≤4 days), last-start × short-rest, log appearances, goalkeeper flag,
never-started flag. Test (matchdays 23-38): log loss 0.400 vs 0.461 for the
typical-rate heuristic; top-11 precision 0.798 vs 0.742. This component is
validated.

P(start) = P(start | available) × P(available). XIs are sampled with Gumbel
top-k on start log-odds: one goalkeeper; line caps of the team's usual shape
+ 1; and a back line kept at its usual size when enough defenders are available,
so defenders are replaced like-for-like while attackers can cover for
attackers. An official XI replaces the sample entirely. The dashboard's
player-impact table removes one likely starter at a time, using the same sampler
with common random numbers.

## 4. News → availability

Extractors (OpenAI structured outputs or deterministic rules) emit
`ExtractedEvent`s with no probability fields. Availability beliefs come from a
fixed table (e.g. unavailable + confirmed → 0.02, unavailable + reported →
0.12, doubtful → 0.45–0.6) or FPL's own chance-of-playing percentage. Votes are
weighted by publisher credibility × freshness (36 h half-life) × certainty
(1.0 / 0.7 / 0.35 / 0.25). Each syndication cluster counts once, and the most
recent confirmed statement dominates. Conflicts are flagged.

## 5. Monte Carlo

For N draws (default 2000): sample both XIs, sample b from its Laplace
posterior, shift the baseline rates, build the score grid. The forecast is the
mean grid, so H + D + A = 1 exactly. Reported uncertainty is the 5–95%
range of per-draw win probabilities and the Monte Carlo standard error.

## 6. Out of scope / not modelled

Cup and European fixtures (no calibrated model: refused). In-play
probabilities. Player identity across clubs (effects are not transferred).
Tactical/formation news are surfaced as evidence but not quantified. The model
does not establish causal effects of any player.
