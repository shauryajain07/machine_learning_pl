# MatchSense AI — data-source coverage

| Source | Key | Provides | Notes |
|---|---|---|---|
| Premier League Fantasy (FPL) public endpoints | none | PL fixtures/results/status, squads with position classes, per-fixture starts/minutes/goals, availability flags (status, chance of playing, news text + timestamp) | Published by the Premier League for its fantasy game; not a documented public API. Cached (5 min–6 h) and throttled. History filtered to the player's current club. |
| openfootball/england (GitHub raw) | none | PL fixtures and results | Community-maintained, public domain; fallback when FPL fails. Kickoffs parsed as Europe/London. |
| BBC Sport club RSS | none | Club news headlines + summaries with pubDate | Feeds are offered for reuse under BBC terms; only headline, summary, link and time are used. |
| API-Football (api-sports.io) | `API_FOOTBALL_KEY` | All competitions (UCL, cups), injuries per fixture, official starting XIs | Needed to rule out cup/European fixtures "tonight" and to replace predicted XIs automatically. The injuries feed has no publication time, so the retrieval time is used and labelled. |
| football-data.org | `FOOTBALL_DATA_API_KEY` | Fixtures across PL/UCL/cups | Free tier, 10 req/min. |
| The Guardian Open Platform | `GUARDIAN_API_KEY` | Football journalism with publication times | |
| NewsAPI.org | `NEWSAPI_KEY` | Multi-publisher search | Free developer plan delays articles, so it is not "fresh"; a paid plan is needed for real time. |
| GNews | `GNEWS_API_KEY` | Multi-publisher search | |
| OpenAI | `OPENAI_API_KEY` | Structured extraction (Responses API + Pydantic schema) | Falls back to the rule extractor on absence or failure. |
| Extra RSS | `MATCHSENSE_EXTRA_RSS` | Any club/league feed the user trusts | Official club sites rarely publish documented feeds, so none are hard-coded. |

No scraping of pages that forbid it is performed. Every news item stores its
publication time, retrieval time, publisher, URL and associated team/player.
Items without a parseable publication time are dropped, because the 72-hour
window cannot be enforced for them.

## Credibility priors (weights, not probabilities)

Official lineup 1.0 · club official 0.95 · FPL 0.9 · BBC / Guardian 0.85 ·
API-Football injuries 0.85 · tier-1 press 0.8 · national press 0.65 ·
aggregators 0.5 · tabloids 0.4 · unknown 0.35. These are editorial
assumptions, documented here and in `providers/base.py`.

## Demo data

`src/matchsense/demo_data/fpl_recording.json` is a recorded snapshot of the FPL
endpoints (2026-11-04) for six clubs. `news_simulated.json` and
`lineups_simulated.json` are **simulated**: fictional publishers on `.invalid`
domains, marked `is_simulated` and shown with a "simulated" badge.
