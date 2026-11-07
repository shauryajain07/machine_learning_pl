"""MatchSense AI: news-aware Premier League match forecasting.

The package wraps the repository's existing recency-weighted Poisson
``ScoreModel`` and ``SeasonState``; it never replaces their fitted logic.
News and lineups only enter through a separately validated lineup-impact
model that shifts the baseline goal rates.
"""

__version__ = "0.1.0"
