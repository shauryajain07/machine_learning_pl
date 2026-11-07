# MatchSense AI: Integrating Structured News Evidence and Probabilistic Lineup Prediction with a Recency-Weighted Poisson Model for Premier League Match Forecasting

**Shaurya Jain**  
*Technical report, November 2026.*

---

## Abstract

We describe and evaluate MatchSense AI, a system that answers natural-language
questions such as *"Will Manchester City win tonight?"* by (i) resolving the
question to a verified fixture in the user's timezone, (ii) retrieving and
structurally interpreting recent football news, (iii) predicting both starting
line-ups probabilistically, and (iv) adjusting an existing, independently
validated recency-weighted Poisson score model through a separately estimated
line-up layer, combined by Monte Carlo simulation. A central design constraint
is that the language model never produces win probabilities: it converts text
into typed, evidence-backed events, which enter the statistical model only
through player-availability beliefs. We evaluate the line-up layer with a
matchday walk-forward protocol on the 2025-26 English Premier League (380
matches; 150 validation and 160 test matches), with strict pre-match information
cut-offs. A pooled logistic start-probability model is clearly useful
out of sample (log loss 0.400 vs 0.461 for a start-rate heuristic; top-11
precision 0.798 vs 0.742). In contrast, line-up effects on goal rates could not
be identified from one season: player-level ridge effects selected on validation
*worsened* held-out log loss (+0.027, 95% bootstrap CI [+0.000, +0.053]), and
no team-level specification improved it with a confidence interval excluding
zero. That includes a position-aware term for a weakened goalkeeper and back
line. We therefore deploy a conservative Bayesian line-up layer whose prior
dominates the posterior, label it provisional in every output, and always show
the unadjusted baseline. We report a post-hoc change to our model-selection
procedure and discuss its consequences for the validity of the test folds.

**Keywords:** football forecasting, Poisson regression, Dixon–Coles, line-up
prediction, information extraction, Monte Carlo simulation, leakage, proper
scoring rules.

---

## 1. Introduction

Statistical models of association-football results have a long history, from
independent Poisson goal models (Maher, 1982) and the low-score correction of
Dixon and Coles (1997) to rating systems such as Elo (Elo, 1978; Hvattum &
Arntzen, 2010) and dynamic or hierarchical extensions (Karlis & Ntzoufras, 2003;
Baio & Blangiardo, 2010; Koopman & Lit, 2015). These models are built from
results and schedule information. However, much pre-match information that
practitioners consider important is not in the results table. It arrives as
unstructured text: injury news, suspensions, press-conference remarks and
predicted or confirmed line-ups.

Large language models (LLMs) make it easy to read such text, and equally easy to
misuse it. An LLM asked "will City win?" will produce a fluent answer with no
calibrated probabilistic meaning. MatchSense is designed around the opposite
principle. The LLM is an *information-extraction* component whose output schema
contains no probability fields. All probabilities come from fitted statistical
models whose out-of-sample behaviour has been measured, and when the measured
evidence is weak the system says so.

The contributions of this report are:

1. **An architecture** that layers news- and line-up-derived features on top of
   an existing, unchanged baseline match model through an offset generalised
   linear model, so the baseline remains available and auditable.
2. **A leakage-safe evaluation** of line-up information at two information sets
   (official XI at about T−60 min; predicted XI from prior selections) using
   matchday walk-forward folds.
3. **A negative result, reported honestly.** One season of line-up data is
   insufficient to identify line-up effects on goal rates, and the most flexible
   specification overfits. We describe how the deployed system degrades to a
   labelled, conservative prior instead of an untested adjustment.
4. **An end-to-end, tested implementation**: fixture resolution, providers,
   extraction, corroboration, line-up sampling, simulation and a dashboard, with
   31 targeted tests including explicit leakage tests.

## 2. Related work

**Goal-based models.** Maher (1982) modelled home and away goals as
independent Poisson variables with team attack and defence parameters. Dixon and
Coles (1997) added a dependence correction for low scores and time-weighting of
past matches, which our baseline inherits through recency weights and a
Dixon–Coles grid. Bivariate Poisson (Karlis & Ntzoufras, 2003), Bayesian
hierarchical (Baio & Blangiardo, 2010) and dynamic state-space formulations
(Koopman & Lit, 2015) extend these ideas.

**Ratings.** Elo ratings (Elo, 1978) were shown to be competitive predictors of
football results by Hvattum and Arntzen (2010). Our baseline uses Elo-style
ratings as inputs.

**Player contribution.** Regularised adjusted plus-minus models estimate
individual contributions by ridge regression on line-up indicators (Hoerl &
Kennard, 1970; Sill, 2010). Kharrat, McHale and Peña (2020) adapted plus-minus
ratings to soccer and stressed the need for heavy regularisation and multiple
seasons of data. Our failure to identify player effects from one season is
consistent with that observation.

**Forecast evaluation.** We use strictly proper scoring rules (Gneiting &
Raftery, 2007): logarithmic loss, the Brier score (Brier, 1950) and the ranked
probability score (Epstein, 1969), which Constantinou and Fenton (2012) argue is
appropriate for ordered football outcomes. Rolling-origin evaluation follows
Tashman (2000). The risks of selection bias when the same data guide model
choice and performance estimation are analysed by Cawley and Talbot (2010).
Leakage in predictive modelling is formalised by Kaufman et al. (2012).

**Sampling and uncertainty.** Line-ups are sampled with the Gumbel-top-k trick
(Kool, van Hoof & Welling, 2019), approximating conditional Bernoulli sampling
(Chen, Dempster & Liu, 1994). Parameter uncertainty uses a Laplace
approximation (Tierney & Kadane, 1986). Confidence intervals for paired score
differences use the bootstrap (Efron & Tibshirani, 1993). Near-duplicate
news detection uses word shingles (Broder, 1997).

## 3. Data

**Historical results.** 9,880 completed Premier League matches (2000-01 to
2025-26) from the openfootball/england repository, ingested and validated by the
project's existing pipeline (explicit team aliases, duplicate and impossible-score
rejection).

**Historical line-ups.** Only the 2025-26 season has full line-up files locally
(`1-premierleague-full.txt`): 380 matches with 11 parsed starters per side,
substitution minutes and goal scorers. This is the sole training source for the
line-up layer. We exclude team-match observations with fewer than three prior
matches in the season, because a "usual XI" is undefined. That leaves 700 of
760 team-match observations.

**Live sources.** Current-season fixtures, results, squads, per-player
starts/minutes/goals and the league's structured availability flags (status,
chance of playing, news text and its timestamp) come from the Premier League's
public fantasy (FPL) endpoints. News comes from BBC Sport club RSS feeds and,
when keys are configured, the Guardian Open Platform, NewsAPI and GNews.
API-Football and football-data.org provide cup and European fixtures, injuries
and official XIs. Coverage, licensing notes and credibility priors are listed in
`docs/DATA_SOURCES.md`.

**What is not available.** There is no archive of historical *news* for the
evaluation season. The news→availability channel therefore cannot itself be
backtested; only the line-up channel it feeds can (Section 6.4).

## 4. Methodology

### 4.1 Baseline match model (unchanged)

The baseline is the repository's deployed `ScoreModel`. Home and away goals
are modelled with Poisson GLMs,

$$
\log \lambda_{h} = \mathbf{x}^\top \boldsymbol\beta_h,\qquad \log \lambda_{a} = \mathbf{x}^\top \boldsymbol\beta_a,
$$

fitted with L2 penalty $\alpha = 0.03$ and exponential recency weights
$w = 2^{-\text{age}/1460\,\text{days}}$. The 32 pre-match features $\mathbf{x}$
include Elo ratings and difference, exponentially weighted scoring and
conceding rates (overall and by venue), points and goal difference per game,
five-match form, draw tendency, rest-day difference, promotion flags and
matchday phase. The score distribution is a 21×21 Dixon–Coles grid. Validation
selected $\rho = 0$ and a 0% blend with a multinomial-logistic outcome model. On
the project's own held-out seasons the baseline attains log loss 0.977 (2024-25)
and 1.027 (2025-26), against 1.085 and 1.082 for a class-frequency prior and
1.005 and 1.033 for a fixed Elo model.

For live use MatchSense copies the bundled `SeasonState` at the end of 2025-26,
calls the existing `start_season` transition (regression to the mean and
promoted-team priors) and applies completed 2026-27 results in date batches
through the existing `update` transition. A result counts as known only if
kickoff + 2 h 30 min precedes the prediction time. The multinomial-logistic
model (C = 0.1, 1460-day half-life) is refitted on the bundled features and
displayed for comparison only.

### 4.2 Query and fixture resolution

Team mentions are resolved against an explicit alias table whose canonical names
match the fitted model state. Bare "City" is read as Manchester City *with an
ambiguity notice* listing the other City clubs. Relative dates are resolved in
the user's IANA timezone. Because 20:00 UK kickoffs fall at 04:00 in Hong Kong,
"tonight" is defined as 17:00–06:00 local time, and the resolved window is
echoed back. Fixtures from all configured providers are merged with source
priority. The system never synthesises a fixture: if no fixture falls in the
window, it reports that and offers the next confirmed one. Fixtures that are in
progress, finished, postponed or in competitions without a calibrated model
(e.g. the UEFA Champions League) receive a status response, not a probability.
This avoids silently transferring a Premier-League-calibrated model across a
distribution shift.

### 4.3 Evidence extraction and corroboration

Items from the preceding 72 hours are clustered into syndication groups when
they share a canonical URL, have word-trigram Jaccard similarity ≥ 0.5, or have
title-token overlap ≥ 0.85 within 48 hours. Each cluster counts once. An
extractor maps each story to typed events:

* entity, team, event type (injury, suspension, possible absence, return…);
* category;
* **certainty** ∈ {confirmed fact, reported expectation, speculation, AI
  inference};
* availability ∈ {available, doubtful, unavailable, unknown};
* match relevance, tactical role, and a supporting quote.

The extractor is either an LLM with a schema-constrained (Pydantic) structured
output, or a deterministic rule extractor used offline and as a fallback.
Mentions are bound to squad players by an identity resolver that handles
initials, accents and name particles and refuses ambiguous matches.

Event votes are combined into a per-player availability belief
$\pi_i \in [0,1]$. Each vote's value comes from a fixed, documented table
indexed by (availability, certainty), or from the league's published
chance-of-playing percentage. Its weight is
$w = c_\text{source} \cdot 2^{-\text{age}/36\text{h}} \cdot c_\text{certainty}$,
and the belief is the weighted mean. The most recent confirmed statement
dominates older reports. Disagreements above 0.4 among non-negligible votes are
flagged as conflicts. These quantities are evidence weights, not forecasts.

### 4.4 Start-probability (selection) model

For each squad player, we estimate $p_i = P(\text{start}_i \mid \text{available})$
with a pooled logistic regression on features computed only from the team's
earlier matches:

* exponentially weighted start rate $\tau_i$ (half-life four matches);
* started last match;
* starts in the last three matches;
* weighted minutes share;
* a short-rest indicator (≤ 4 days) and its interaction with "started last";
* log appearances;
* goalkeeper and never-started flags.

The regularisation strength C was chosen on matchdays 16–22 from
{0.03, 0.1, 0.3, 1, 3}. The live start probability is $p_i \pi_i$. Valid XIs
are sampled by perturbing log-odds with Gumbel noise and greedily selecting one
goalkeeper and ten outfielders, subject to line caps of the team's usual shape
+ 1. The back line is kept at its usual size whenever enough defenders are
available, so defenders are replaced like-for-like while attackers may cover for
attackers. When an official XI is published it replaces the sampled line-ups
entirely.

### 4.5 Line-up impact model

Let $s_i \in \{0,1\}$ indicate that player $i$ starts. For team $t$ facing $o$
we define two pre-match line-up features:

$$
D_t = 1 - \frac{\sum_i \tau_i s_i}{\sum_{\text{top }11} \tau_i}, \qquad
G_t = \sum_i g_i (1 - s_i),
$$

* $D_t$, *discontinuity*, is the share of the team's usual starting mass absent
  from the XI.
* $G_t$, *absent goal share*, uses $g_i$, the player's share of the team's
  decayed recent goals.
* $D^{\text{def}}_o$, *defensive discontinuity*, is $D$ restricted to the
  opponent's goalkeeper and back line and to its top five by $\tau$. Training
  labels come from the Football.TXT line-up layout: group 0 is the goalkeeper
  and group 1 the back line, which gives one keeper and 3–5 defenders in all
  760 team line-ups. Live labels come from FPL positions.

Goals are modelled as a Poisson GLM with the baseline as an offset (McCullagh &
Nelder, 1989):

$$
\log \mathbb E[y_t] = \log \lambda_t^{\text{base}} + b_1 \tilde D_t + b_2 \tilde D_o + b_3 \tilde G_t + b_4 \tilde G_o + b_5 \tilde D^{\text{def}}_o
\;\Big[ + \sum_i a_i (s_i - \tau_i) + \sum_j d_j (s_j - \tau_j) \Big],
$$

where tildes denote centring at training means, so an average amount of
rotation leaves the baseline unchanged. The bracketed player-level terms (a
plus-minus analogue) are included only in some candidates. Coefficients are
MAP estimates under Gaussian priors (ridge penalties), with Laplace covariance
$\Sigma = (X^\top \operatorname{diag}(\mu) X + \Lambda)^{-1}$.

Candidates (21 in total):

* the baseline;
* continuity only, $\alpha \in \{1, 5, 20, 80\}$;
* continuity plus goal share, $\alpha \in \{5, 20, 80\}$;
* continuity plus defence, $\alpha \in \{5, 20, 80\}$;
* continuity plus goal share plus defence, $\alpha \in \{20, 80\}$;
* continuity plus player effects, $\alpha \in \{5, 20\}$ and
  $\alpha_p \in \{10, 30, 100, 300\}$.

A **provisional prior** model uses all five team features, with informative
priors of means $(-0.15, +0.10, -0.25, 0, +0.20)$ and standard deviations
$(0.15, 0.15, 0.15, 0.10, 0.15)$. These encode three domain assumptions:

* losing all of a side's recent scorers reduces its scoring rate by roughly
  22%, because replacements recover most output;
* a fully changed XI reduces it by roughly 14%;
* an opponent missing its whole usual goalkeeper and back line concedes roughly
  22% more.

They are assumptions, not estimates.

### 4.6 Monte Carlo forecast

For $n = 1, \dots, N$ (default N = 2000):

1. sample both XIs (§4.4);
2. sample $\mathbf b^{(n)} \sim \mathcal N(\hat{\mathbf b}, \Sigma)$;
3. compute $\lambda^{(n)}_{h,a} = \lambda^{\text{base}}_{h,a} \exp(\Delta\eta^{(n)}_{h,a})$;
4. form the Dixon–Coles grid $P^{(n)}$.

The forecast is $\bar P = N^{-1} \sum_n P^{(n)}$. Home/draw/away probabilities,
expected goals and scorelines all derive from $\bar P$, so the probabilities sum
to one exactly. Uncertainty is reported as the 5–95% range of the per-draw win
probabilities and the Monte Carlo standard error. For transparency, every draw
is returned to the user along with one scoreline sampled from $P^{(n)}$: the
per-draw probabilities, goal rates and simulated result. The headline
probabilities remain the exact average $\bar P$, not the noisier share of
sampled results. Per-player contributions are
leave-one-out differences in expected goals when one player's availability
evidence is removed. They are explicitly described as model associations, not
causal effects.

## 5. Experimental design

**Information sets.** (B) baseline only; (C) baseline + official XI, public at
about T−60 min; (P) baseline + XI predicted from earlier selections only, i.e.
roughly what is known the day before minus news.

**Splits.** Matchday walk-forward on 2025-26 (Tashman, 2000). For fold *k*,
models are fitted on every eligible observation that kicked off strictly before
the earliest kickoff of matchday *k*, which also handles rearranged fixtures.
Validation folds are matchdays 8–22 (150 matches) and test folds are matchdays
23–38 (160 matches). The baseline rates are produced by the existing
`ScoreModel` fitted through 2024-25 (trained through 2025-05-25), with
in-season features from earlier dates only.

**Metrics.** Log loss (primary), multiclass Brier score, RPS, accuracy, and
exact-score negative log-likelihood. We also report reliability tables and
paired bootstrap 95% intervals for per-match log-loss differences (4,000
resamples).

**Selection rule.** Round 1 selected the candidate with the lowest validation
log loss. Rounds 2 and 3 (see §7.2) applied the one-standard-error rule (Hastie,
Tibshirani & Friedman, 2009): the simplest candidate within one paired standard
error of the best. Round 3 added the defensive candidates.

**Leakage controls.** We check leakage in four ways (Kaufman et al., 2012):

1. Training sets are asserted to end before each fold's cut-off.
2. Line-up state is snapshotted before the match is applied, and a test rewrites
   every future line-up, goal and result and asserts that all earlier features
   are bit-identical.
3. Baseline roll-forward ignores results whose kickoff + 2.5 h is after the
   prediction time, and a test mutates such a result and asserts identical
   probabilities.
4. News and availability flags published after the prediction time are excluded,
   and a test checks that a later confirmation changes the belief only once the
   clock passes its publication time.

## 6. Results

### 6.1 Start-probability model

| Model | Test log loss | Top-11 precision |
|---|---:|---:|
| Typical start-rate heuristic | 0.461 | 0.742 |
| Logistic selection model (C = 3) | **0.400** | **0.798** |

Test set: 8,493 player-match rows from matchdays 23–38. Validation log loss
across C ranged from 0.442 (C = 0.03) to 0.432 (C = 3). Normalising
probabilities to exactly eleven starters per team changed test log loss only
from 0.400 to 0.404.

### 6.2 Line-up impact candidates

Table 2. Walk-forward results, final round (lower is better).

| Candidate | Val. log loss | Val. Brier | Test log loss | Test Brier | Test RPS |
|---|---:|---:|---:|---:|---:|
| Baseline (B) | 1.0322 | 0.6220 | 1.0498 | 0.6338 | 0.2146 |
| Continuity + defence, α = 20 (C) | 1.0319 | 0.6218 | 1.0495 | 0.6331 | 0.2142 |
| Continuity + defence, α = 5 (C; one-SE choice) | 1.0320 | 0.6217 | 1.0498 | 0.6330 | 0.2140 |
| Continuity + goal share + defence, α = 80 (C) | 1.0322 | 0.6220 | 1.0499 | 0.6336 | 0.2145 |
| Player effects, α = 20, α_p = 100 (C) | **1.0283** | **0.6189** | 1.0560 | 0.6371 | 0.2160 |
| Provisional prior (C, deployed) | 1.0295 | 0.6205 | 1.0517 | 0.6349 | 0.2148 |

Paired test differences against the baseline (95% bootstrap CI):

* round-2 one-SE choice (player effects, α = 5, α_p = 30): **+0.0267
  [+0.0004, +0.0533]**, significantly *worse* than the baseline;
* round-3 one-SE choice (continuity + defence, α = 5): +0.00003
  [−0.0051, +0.0053];
* deployed provisional prior: +0.0019 [−0.0057, +0.0096];
* provisional prior under information set (P): −0.0004 [−0.0033, +0.0025].

### 6.3 Posterior of the deployed layer

| Coefficient | Prior mean (sd) | Posterior mean ± sd |
|---|---:|---:|
| $b_1$ own discontinuity | −0.15 (0.15) | −0.150 ± 0.137 |
| $b_2$ opponent discontinuity | +0.10 (0.15) | −0.009 ± 0.138 |
| $b_3$ own absent goal share | −0.25 (0.15) | −0.187 ± 0.116 |
| $b_4$ opponent absent goal share | 0.00 (0.10) | −0.008 ± 0.087 |
| $b_5$ opponent defensive discontinuity | +0.20 (0.15) | +0.046 ± 0.126 |

Posterior standard deviations are only slightly smaller than prior standard
deviations, so the data are weakly informative. The defensive coefficient is
the exception: the data pulled it from +0.20 to +0.05, which suggests a broken
back line has a smaller effect in this sample than assumed. With uninformative
(pure ridge) priors, every team-level coefficient lay within about one
posterior standard deviation of zero (e.g. $b_3 = -0.059 \pm 0.113$ at α = 20).

### 6.4 Missing-data analysis

On the test folds we replaced a random share of players' line-up indicators by
their typical start rates, i.e. removed that information. Log loss was 1.0517,
1.0524, 1.0492, 1.0546 and 1.0491 at 0%, 25%, 50%, 75% and 100% hidden. There
is no monotone trend: in this sample, line-up information does not measurably
change accuracy. The system therefore degrades to the baseline without a
measurable cost.

### 6.5 Calibration

Reliability for the home-win probability on the 160 test matches is similar for
both models. In the 0.3–0.4 bin the baseline predicted 0.355 vs 0.405 observed
(n = 37), and the provisional model 0.350 vs 0.361 (n = 36). In the 0.5–0.6 bin
they predicted 0.552 vs 0.533 (n = 30) and 0.552 vs 0.543 (n = 35). Bins
beyond 0.7 contain at most five matches and are not interpretable.

### 6.6 Case study (demonstration data)

The demonstration replays a recorded FPL snapshot (4 November 2026) for
Nottingham Forest v Manchester City, with **simulated** news from fictional
publishers. The simulated evidence contains:

* a wire report that Haaland is a doubt;
* a tabloid report that he will miss the match, plus a syndicated copy, which
  are merged into one cluster;
* a confirmed suspension;
* a later confirmed fitness pass.

At 08:00 UTC on match day, Haaland's corroborated availability was 0.31
(doubtful), and City's win probability moved from 55% (baseline) to 52%. The
top contribution was Haaland's availability, worth −0.10 expected goals. The
what-if comparison (Haaland available vs unavailable) changed City's win
probability by −3.7 points. At 12:00 the simulated confirmation became visible,
the belief rose to 0.98 and the forecast returned to the baseline. After the
simulated official XIs were released at T−60 min, both predicted line-ups were
replaced by confirmed ones and the forecast was recomputed (56%).

In live mode on 4 November 2026 the question "Will Manchester City win tonight?"
correctly returned *no fixture tonight* (17:00–06:00 HKT) and offered Nottingham
Forest v Manchester City on 8 November 01:30 HKT. It also stated that cup and
European fixtures could not be ruled out without a fixtures API key.

## 7. Discussion

### 7.1 Why line-up effects were not identified

Three factors explain the null result:

1. **Sample size.** 700 team-match observations is small relative to the
   variance of football scores. Kharrat et al. (2020) needed multiple seasons for
   stable plus-minus estimates.
2. **Confounding by selection.** Managers rotate against weaker opponents and in
   congested periods, so line-up discontinuity is correlated with match context
   that the baseline already partly captures. Coefficients are therefore
   associational.
3. **Overlap with the baseline.** Form features already absorb persistent
   absences, which leaves only transient deviations for the line-up layer.

The most flexible model improved validation log loss by 0.004 while losing 0.027
on test. This is a textbook instance of the selection-induced optimism
described by Cawley and Talbot (2010).

### 7.2 Disclosure of a post-hoc change

The first evaluation round selected player-level effects by minimum validation
log loss, and that model degraded the test folds. We then added three things
after viewing the test results: the one-SE rule, the goal-share feature and the
provisional prior. A third round added the opponent defensive-discontinuity
term, after rounds 1–2 had been seen. It was requested because the deployed
model gave defenders no defensive value, so absences of defenders could even
*raise* their team's win probability when an attacker replaced them. The
round-3 sampler change (like-for-like defensive replacement) has the same
status. Consequently, **the test folds are not an untouched holdout
for the deployed configuration**, and the neutral result reported for the
provisional prior should be read as a sanity check, not a confirmatory estimate.
The round-1 and round-2 reports are preserved in the repository
(`REPORT_round1.md`, `REPORT_round2.md`).

### 7.3 Why deploy a prior at all

The user-facing question concerns news, and some news, such as a confirmed
absence of a team's leading scorer, is plausibly informative. We chose to
encode a weak, explicitly stated prior, let one season of data update it, and
label the result. The alternative was to ignore news entirely, or to deploy
coefficients that measurably overfit. The unadjusted baseline is always
displayed next to the adjusted forecast, so users can discount the adjustment
entirely. No claim is made that news improves accuracy; Section 6 shows that
this has not been demonstrated.

## 8. System validation

The implementation (`src/matchsense/`) contains 31 targeted tests on top of the
project's existing 57, all passing. They cover:

* team-alias resolution and timezone windows;
* "tonight" with no fixture;
* conflicting and syndicated reports;
* precedence of confirmed statements;
* the LLM structured-output contract (no probability fields);
* exclusion of injured players and validity of eleven-player XIs;
* replacement of predicted XIs by official ones;
* player identity resolution, including refusal on ambiguity;
* probabilities summing to one;
* adjusted-vs-baseline behaviour;
* missing credentials;
* cache retry and stale fallback;
* leakage tests for line-up features, baseline roll-forward and news timing;
* in-progress matches and unsupported competitions;
* defensive discontinuity and like-for-like replacement of defenders;
* the LLM query interpreter (Codex CLI) contract, validation, caching and
  fallback.

## 9. Limitations and threats to validity

**Construct validity.** FPL position classes are coarse, so formations are
approximate. Individual defensive quality is not estimated: defenders matter
only through how established they are and whether the usual back line is
broken.

**Internal validity.** We used one season with post-hoc design changes (§7.2).
The availability value table and credibility weights are assumptions, not
estimates.

**External validity.** The model covers the Premier League only and refuses
other competitions. Player effects are not transferred across clubs.

**Evaluation gaps.** The news channel is untested historically because no news
archive exists for the season. Calibration bins at extreme probabilities are
sparse.

**Data provenance.** The FPL endpoints are public but undocumented. Free news
tiers may be delayed (e.g. NewsAPI's developer plan).

## 10. Conclusion

MatchSense shows how unstructured football news can be connected to a
validated statistical forecaster without letting a language model invent
probabilities. Evidence is typed, sourced, time-stamped and corroborated, and
it enters only through player availability and line-up simulation. The
evaluation separates what works from what does not. Start-probability modelling
is useful out of sample. Line-up effects on goal rates could not be identified
from one season, and the most expressive specification overfits. Future work
should:

* add multiple seasons of line-ups, ideally with event-level data such as xG,
  to estimate player effects with hierarchical shrinkage;
* archive timestamped news prospectively so that the news channel itself can be
  evaluated without leakage;
* re-evaluate on a genuinely untouched season.

## References

Baio, G., & Blangiardo, M. (2010). Bayesian hierarchical model for the prediction of football results. *Journal of Applied Statistics, 37*(2), 253–264.

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3.

Broder, A. Z. (1997). On the resemblance and containment of documents. In *Proceedings of Compression and Complexity of Sequences* (pp. 21–29). IEEE.

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

Chen, X.-H., Dempster, A. P., & Liu, J. S. (1994). Weighted finite population sampling to maximize entropy. *Biometrika, 81*(3), 457–469.

Constantinou, A. C., & Fenton, N. E. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports, 8*(1).

Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C (Applied Statistics), 46*(2), 265–280.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco.

Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology, 8*(6), 985–987.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67.

Hvattum, L. M., & Arntzen, H. (2010). Using ELO ratings for match result prediction in association football. *International Journal of Forecasting, 26*(3), 460–470.

Karlis, D., & Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models. *Journal of the Royal Statistical Society: Series D (The Statistician), 52*(3), 381–393.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), 15.

Kharrat, T., McHale, I. G., & Peña, J. L. (2020). Plus–minus player ratings for soccer. *European Journal of Operational Research, 283*(2), 726–736.

Kool, W., van Hoof, H., & Welling, M. (2019). Stochastic beams and where to find them: The Gumbel-top-k trick for sampling sequences without replacement. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR 97, pp. 3499–3508).

Koopman, S. J., & Lit, R. (2015). A dynamic bivariate Poisson model for analysing and forecasting match results in the English Premier League. *Journal of the Royal Statistical Society: Series A, 178*(1), 167–186.

Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica, 36*(3), 109–118.

McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.

Sill, J. (2010). Improved NBA adjusted +/− using regularization and out-of-sample testing. *MIT Sloan Sports Analytics Conference*.

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting, 16*(4), 437–450.

Tierney, L., & Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal densities. *Journal of the American Statistical Association, 81*(393), 82–86.

---

### Appendix A. Reproducibility

```bash
./.venv/bin/python -m src.scripts.train_season_simulator --source local      # baseline (existing)
./.venv/bin/python -m src.scripts.train_lineup_impact                        # §5–6
./.venv/bin/python -m pytest                                                  # §8
```

Artifacts: `artifacts/matchsense/{lineup_impact.json, metrics.json, REPORT.md, REPORT_round1.md}`.
Software: Python 3.15, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.4.
Random seeds: walk-forward bootstrap 7; missing-data masks 11; Monte Carlo 20261104.

### Appendix B. Data availability

openfootball/england (public domain); Premier League FPL public endpoints;
BBC Sport RSS (headline, summary, link and time only). The demonstration news
and official line-ups are simulated and must not be cited as factual.
