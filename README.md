# MatchSense AI — news-aware Premier League forecasting

Ask *"Will Manchester City win tonight?"* and MatchSense resolves the real
fixture in your timezone (default Asia/Hong_Kong), gathers the last 72 hours of
evidence for both clubs, predicts both starting XIs, runs **this repository's
existing Poisson `ScoreModel`** as the baseline, and adjusts it through a
separately evaluated lineup layer with Monte Carlo simulation. If there is no
match tonight it says so and offers the next confirmed fixture. The AI only
converts text into structured evidence; it never produces probabilities.

Research write-up: [`docs/paper/MatchSense_AI.md`](docs/paper/MatchSense_AI.md) ·
model assumptions: [`docs/MODEL.md`](docs/MODEL.md) · sources:
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Quick start

```bash
python3.15 -m venv .venv          # any Python >= 3.11
./.venv/bin/pip install -e '.[dev,llm]'
cp .env.example .env              # optional keys; everything degrades gracefully
```

The baseline bundle must exist (`artifacts/season_simulator/model.joblib`, from
`train_season_simulator`). Train the lineup layer (about 10 s):

```bash
./.venv/bin/python -m src.scripts.train_lineup_impact
```

Run the app:

```bash
./.venv/bin/python -m src.scripts.serve           # live mode → http://127.0.0.1:8000
```

```bash
./.venv/bin/python -m src.scripts.serve --demo    # offline demo → http://127.0.0.1:8001
```

Tests (88, offline):

```bash
./.venv/bin/python -m pytest
```

Refresh the recorded demo snapshot (network):

```bash
./.venv/bin/python -m src.scripts.record_demo_data
```

## What works without keys

Live mode works with **no API keys**: fixtures, results, squads, per-player
selection history and official availability flags come from the Premier
League's public FPL endpoints; news from BBC Sport club RSS; extraction from the
deterministic rule extractor. Add keys in `.env` for: cup/European fixtures,
injuries and official XIs (`API_FOOTBALL_KEY`, `FOOTBALL_DATA_API_KEY`), more
journalism (`GUARDIAN_API_KEY`, `NEWSAPI_KEY`, `GNEWS_API_KEY`) and LLM
extraction (`OPENAI_API_KEY`, model via `OPENAI_MODEL`). Without a fixtures key,
answers state that cup/European matches could not be ruled out.

Demo mode replays a **recorded** FPL snapshot (2026-11-04) and **simulated**
news and official XIs (fictional publishers, `.invalid` URLs, labelled in the
UI). A demo-clock selector replays three moments: before lineups, after the
Forest v City XIs are released, and a midweek "no fixture tonight".

## Query understanding (Codex CLI)

Every question is first read by an LLM interpreter. By default this is the
Codex CLI (`codex exec` with a strict JSON schema), falling back to the OpenAI
API, then to the rule parser. It returns intent, clubs, a verbatim date phrase
and players in/out. Clubs are validated against the registry, dates are
re-resolved deterministically in your timezone, and players are matched to real
squads. Results are cached per question in `data/query_interpretations.sqlite`.
A new question through Codex costs about 20k tokens and 15 s; a repeat costs
nothing. Choose the backend with `MATCHSENSE_QUERY_INTERPRETER`. News
extractions are cached the same way in `data/llm_extractions.sqlite`.

## Architecture

```
src/matchsense/
  interpreter.py     Codex CLI / OpenAI query understanding (validated, cached, rule fallback)
  query.py           NL parsing: teams/aliases, intent, timezone-aware windows, what-if players
  teams.py           canonical clubs (same names as the fitted SeasonState), aliases, provider spellings
  http.py            SQLite response cache, retries/backoff, per-host throttling, stale fallback
  providers/         fpl.py · fixtures_extra.py (API-Football, football-data, openfootball) ·
                     news.py (BBC RSS, Guardian, NewsAPI, GNews, extra RSS) · demo.py
  intelligence.py    syndication dedup, LLM/rule extraction (Pydantic), corroboration → availability
  players.py         player identity resolution (initials, accents, particles, ambiguity refusal)
  lineup_data.py     leakage-safe selection state shared by training (openfootball) and live (FPL)
  impact.py          lineup-impact GLM (offset Poisson, MAP/Laplace) + selection model
  training.py        observations, walk-forward folds, one-SE rule, bootstrap
  lineups.py         P(start), constrained XI sampling, confirmed-XI override
  baseline.py        existing ScoreModel + SeasonState roll-forward; logistic comparison
  simulate.py        Monte Carlo over XIs and coefficient posterior → score grid
  pipeline.py        orchestration, status handling, explanations, what-if
  storage.py         SQLite prediction history + outcome tracking and scoring
  api.py, static/    FastAPI + vanilla-JS dashboard (no build step)
```

The dashboard has no build step: the repository had no frontend, and a single
static bundle served by FastAPI is the least maintenance for a Python ML
project.

## Limitations (read before trusting a number)

* **Lineup/news adjustments are provisional.** One season of lineup data could
  not identify lineup effects out of sample; the deployed layer is a
  conservative prior updated with that data, and is labelled as such. News
  adjustments themselves cannot be backtested (no historical news archive).
* The model is calibrated on the Premier League only. Other competitions get
  fixture details but no probabilities.
* Pre-match only: no in-play forecasts.
* FPL position classes are coarse (wingers are "MID"), so formations are approximate.
* FPL endpoints are public but undocumented; they may change without notice.
* Prediction history scores only forecasts issued before kickoff; small samples
  say nothing about accuracy.

---

# Premier League historical database (data and baseline models)

This project ingests the local `openfootball/england` repository into a normalized Supabase PostgreSQL database. It stores match facts only: no standings, champions, Elo, rolling form, ratings, feature engineering, or prediction models are included.

The importer prefers `<season>/1-premierleague-full.txt`. It falls back to `<season>/1-premierleague.txt` unless `--full-only` is supplied. Compact files contribute only the facts they contain; missing venues, UTC offsets, lineups, substitutions, cards, and officials remain `NULL`/empty.

## Setup

1. Apply the SQL migration to Supabase:

   ```bash
   python -m src.scripts.migrate_supabase --db-url "$SUPABASE_DB_URL"
   ```

   The same SQL can be pasted into the Supabase SQL editor.

2. Install the project:

   ```bash
   python -m pip install -e '.[dev]'
   ```

3. Clone the source repository locally, then run a dry run:

   ```bash
   git clone https://github.com/openfootball/england.git ./england
   python -m src.scripts.ingest_openfootball ./england --all-seasons --dry-run --verbose
   ```

4. Persist the records to Supabase:

   ```bash
   python -m src.scripts.ingest_openfootball ./england --all-seasons \
     --db-url "$SUPABASE_DB_URL"
   ```

   `SUPABASE_DB_URL` and `DATABASE_URL` are also read automatically. A Supabase pooler URL is usually the most reliable option for a bulk import.

   When a direct database URL is unavailable, generate bounded SQL batches and submit them with the authenticated Supabase MCP `execute_sql` tool. This path uses the same parser and normalization code and is idempotent:

   ```bash
   python -m src.scripts.generate_mcp_batches ./england /tmp/pl-mcp-batches \
     --run-token "<unique-run-token>" \
     --source-commit-sha "<pinned-commit-sha>"
   ```

   Submit the generated files in manifest order. Do not use a public `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` for schema or import writes.

## Useful commands

```bash
python -m src.scripts.ingest_openfootball ./england --season 2025-26 --dry-run
python -m src.scripts.ingest_openfootball ./england --all-seasons --full-only
python -m src.scripts.inspect_database --db-url "$SUPABASE_DB_URL"
pytest
```

The source commit SHA and file SHA-256 are stored in `ingestion_runs` and `source_files`. Each match retains an exact `raw_match_text` block, its source path, source index, deterministic fingerprint, and parser status. Child rows retain raw event/player text and are replaced transactionally when their source match is re-ingested, so corrections update the same match rather than creating duplicates.

## ML baseline: football Elo

The first predictive layer is a dependency-free, chronological football Elo
backtest. It tracks every canonical club in the historical Premier League
data, applies season regression, home advantage, a draw-aware probability
model, and a capped goal-margin adjustment. Pre-match features are written
without using the current match result.

Run it against the pinned clone with:

```bash
python -m src.scripts.train_football_elo ./england
```

By default it tunes parameters on `2021-22` through `2023-24`, tests on
`2024-25` and `2025-26`, and excludes the incomplete `2026-27` season. It
writes `metrics.json`, `elo_features.csv`, and the final ratings snapshot to
`artifacts/football_elo/`. The rating is an interpretable baseline and
feature source.

The supervised home/draw/away model adds a leakage-safe pre-match feature
vector. It includes rolling five-match PPG, goals scored, goals conceded, and
goal difference; season-to-date form; home-only and away-only PPG/GD; recent
opponent Elo strength; opponent-adjusted point form; attack-vs-defence
matchups; home/away rest days; matchweek and season progression; and
promoted-team flags. The categorical home and away team identities are also
one-hot encoded. All of these values are generated before the current score is
applied:

```bash
python -m src.scripts.train_match_outcomes ./england
```

It trains multinomial logistic regression on chronological splits and writes
the leakage-safe feature matrix, deployment coefficients, and validation/test
metrics to `artifacts/match_outcomes/`. Neither model changes the raw fact
tables.

The season simulator also derives leakage-safe five-match form, draw tendency,
rest, strength-gap, and combined-goal-rate features. When `--source supabase`
is used, it additionally derives pre-match context from earlier appearances,
cards, substitutions, and referee history. The database loader records these
features only after the current fixture's prediction row has been created, so
the current match's lineup or cards cannot leak into its own prediction.

## Reproducible baseline evaluation

For the current milestone, run the stricter baseline protocol from the pinned
OpenFootball checkout:

```bash
./.venv/bin/python -m src.scripts.evaluate_baselines ./england --verbose
```

The runner canonicalizes one completed row per match, applies explicit team
aliases, rejects missing/impossible results and duplicate fixtures, excludes
incomplete seasons, and derives `H`, `D`, or `A` only from the final score.
It uses all complete seasons before the next season for training, the next
complete season for validation, and the latest complete season as untouched
test data. The default checkout therefore uses `2000-01`–`2023-24` for
training, `2024-25` for validation, and `2025-26` for test; the in-progress
`2026-27` season is excluded.

The two validation baselines are an empirical training-set class-frequency
prior and multinomial logistic regression over pre-match team/form/Elo
features. Numeric imputers/scalers and team encoders are fitted inside each
training window only. Metrics include log loss, multiclass Brier score, ranked
probability score, per-class recall, and upset performance. `--verbose` prints
the optimizer fit size and completed LBFGS iterations in the terminal.

The run writes `canonical_matches.csv`, `pre_match_features.csv`,
`metrics.json`, `data_quality.json`, `leakage_check.json`, predictions, and a
deployment model under `artifacts/baseline_evaluation/`. The leakage check
replaces every result after the validation cutoff, rebuilds the feature stream,
and requires exact equality for all earlier raw and fitted-preprocessor feature
rows.

To evaluate the normalized Supabase data directly, set the pooled connection
string and use:

```bash
./.venv/bin/python -m src.scripts.evaluate_baselines \
  --source supabase --db-url "$SUPABASE_DB_URL" --verbose
```

When the project-scoped Supabase MCP session is available but a PostgreSQL
password is not present in the terminal, export the read-only query result and
run the same evaluator with:

```bash
./.venv/bin/python -m src.scripts.evaluate_baselines \
  --source supabase \
  --supabase-export artifacts/supabase_mcp_export.json \
  --verbose
```

The export is source data only; canonical validation, chronological features,
training-only preprocessing, and the post-cutoff mutation check still run
locally. The current Supabase snapshot contains 9,890 rows, of which 9,880
belong to complete seasons after excluding the incomplete `2026-27` season.

## Walk-forward Poisson versus logistic comparison

To decide whether multinomial logistic probabilities improve the goal-based
Poisson forecast, run:

```bash
./.venv/bin/python -m src.scripts.compare_match_outcomes ./england
```

The runner makes expanding matchweek predictions for the validation and final
test seasons. Each fold is fitted only on matches dated before that fold's
earliest fixture, which is conservative for same-day games and postponements.
It uses the season simulator's current pre-match state features and
recency-weighted `FittedModel` recipe. The defaults match its selected
candidates: Poisson alpha `0.03`, logistic C `0.1`, and a 1,460-day half-life.
The class-frequency prior, two Poisson goal regressors, and multinomial
logistic model are refitted inside each fold. The held-out test season is not
used to fit probability temperatures or select the logistic blend weight.
Previously played test-season matches can enter later chronological training
folds, as they would be available at those later kickoff dates.

Multiclass temperature scaling and the blend weight are fitted only from
walk-forward validation predictions. The blend searches
`P_final=(1-w)P_Poisson+wP_logistic` by validation log loss. Upsets use one
shared, prediction-independent rule: a non-draw win by the lower-rated side
under pre-match Elo including home advantage; draws and exact Elo ties are
excluded.

The run writes `REPORT.md`, the combined `comparison.csv`, separate Poisson
and logistic raw/calibrated prediction files, selected-blend predictions,
metrics, source quality details, and the exact `season_model_features.csv`
input matrix under
`artifacts/walk_forward_comparison/`. It fails if any fold's latest training
match date is not strictly before its prediction date or if any probability
row does not sum to one within `1e-6`.

External numeric pre-match signals can be supplied with `--context context.csv`.
The CSV must contain `season,date,home,away` plus numeric columns; an optional
`available_date` must not be after `date`. Useful columns include market
probabilities and rolling pre-match xG or availability summaries. Same-match
goals, results, and other post-match targets are rejected.

For a top-1 feed that should explicitly include a controlled number of draws,
train with `--draw-focused`. This selects the decision multiplier by validation
macro-F1 and caps the validation draw-pick rate (10% by default); the underlying
probabilities and season simulation remain unchanged.

## Schema highlights

The core tables are `competitions`, `seasons`, `rounds`, `teams`, `team_aliases`, `players`, `player_aliases`, `referees`, `matches`, `match_team_names`, `appearances`, `substitutions`, `goals`, `cards`, `match_referees`, and `parsing_issues`.

`matches.matchday`, `matches.match_date`, `matches.day_of_week`, `matches.kickoff_local_time`, `matches.utc_offset_minutes`, and `matches.kickoff_utc` make schedule analysis straightforward. Goal, card, and substitution rows retain `minute`, `stoppage_time`, `absolute_minute`, and `raw_minute`; `45+2'` is never reduced to only `47`.

The source's explicit `Yellow-Red:` event label is stored as `cards.card_type = 'yellow_red'`; the importer does not guess whether it represents a direct red or a second yellow.

The checked-in team alias file is intentionally conservative and explicit. Edit it when historical names are verified; an alias is never inferred from fuzzy matching. Player names are normalized with Unicode-preserving casefolding, so capitalization changes do not create duplicate player rows while accents remain intact.
