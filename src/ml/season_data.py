"""Read-only, validated inputs for training and season simulation."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from src.db.connection import connect
from src.ingestion.football_txt.header_parser import parse_header, parse_date_parts, resolve_date
from src.ingestion.football_txt.score_parser import SCORE_RE, parse_leading_time
from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.aliases import load_alias_maps
from src.scripts.train_football_elo import load_scored_matches
from src.ml.season_context import MatchContext, fixture_key


@dataclass(frozen=True)
class Match:
    season: str
    date: date
    home: str
    away: str
    hg: int | None = None
    ag: int | None = None
    matchday: int | None = None
    source_id: str = ""

    @property
    def year(self):
        return int(self.season[:4])

    @property
    def outcome(self):
        if self.hg is None or self.ag is None:
            raise ValueError("Unplayed fixture has no target")
        return 0 if self.hg > self.ag else 2 if self.hg < self.ag else 1


# Explicit projection excludes lineups, events and other post-match features.
EXPORT_SQL = """
select s.season_label as season, m.match_date as date,
       h.canonical_name as home, a.canonical_name as away,
       m.home_score as hg, m.away_score as ag, m.matchday,
       m.match_id::text as source_id
from public.matches m
join public.seasons s on s.season_id = m.season_id
join public.competitions c on c.competition_id = s.competition_id
join public.teams h on h.team_id = m.home_team_id
join public.teams a on a.team_id = m.away_team_id
where c.name = %s and c.country = %s and s.season_label < %s and m.parse_status <> 'failed'
order by s.season_label, m.match_date, m.match_id
"""

# These queries intentionally return only facts known in the source before the
# model is fit.  ``build_database_context`` converts each match's event facts
# into features using only earlier matches, so a lineup/card from the target
# match can never leak into its own prediction.
DATABASE_TEAMS_SQL = """
select m.match_id::text as source_id, m.home_team_id::text as home_id,
       m.away_team_id::text as away_id
from public.matches m
join public.seasons s on s.season_id = m.season_id
join public.competitions c on c.competition_id = s.competition_id
where c.name = %s and c.country = %s and s.season_label < %s
  and m.parse_status <> 'failed'
"""

DATABASE_APPEARANCES_SQL = """
select m.match_id::text as source_id, m.home_team_id::text as home_id,
       m.away_team_id::text as away_id, a.team_id::text as team_id,
       a.player_id::text as player_id, a.started
from public.matches m
join public.seasons s on s.season_id = m.season_id
join public.competitions c on c.competition_id = s.competition_id
join public.appearances a on a.match_id = m.match_id
where c.name = %s and c.country = %s and s.season_label < %s
  and m.parse_status <> 'failed'
order by m.match_date, m.match_id, a.source_order
"""

DATABASE_CARDS_SQL = """
select m.match_id::text as source_id, m.home_team_id::text as home_id,
       m.away_team_id::text as away_id, card.team_id::text as team_id,
       card.card_type
from public.matches m
join public.seasons s on s.season_id = m.season_id
join public.competitions c on c.competition_id = s.competition_id
join public.cards card on card.match_id = m.match_id
where c.name = %s and c.country = %s and s.season_label < %s
  and m.parse_status <> 'failed'
order by m.match_date, m.match_id, card.source_order
"""

DATABASE_SUBSTITUTIONS_SQL = """
select m.match_id::text as source_id, m.home_team_id::text as home_id,
       m.away_team_id::text as away_id, sub.team_id::text as team_id
from public.matches m
join public.seasons s on s.season_id = m.season_id
join public.competitions c on c.competition_id = s.competition_id
join public.substitutions sub on sub.match_id = m.match_id
where c.name = %s and c.country = %s and s.season_label < %s
  and m.parse_status <> 'failed'
order by m.match_date, m.match_id, sub.source_order
"""

DATABASE_REFEREES_SQL = """
select m.match_id::text as source_id, m.home_team_id::text as home_id,
       m.away_team_id::text as away_id, mr.referee_id::text as referee_id
from public.matches m
join public.seasons s on s.season_id = m.season_id
join public.competitions c on c.competition_id = s.competition_id
join public.match_referees mr on mr.match_id = m.match_id
where c.name = %s and c.country = %s and s.season_label < %s
  and m.parse_status <> 'failed'
order by m.match_date, m.match_id, mr.referee_order
"""

DATABASE_CONTEXT_FEATURES = (
    'home_prior_lineup_size', 'away_prior_lineup_size',
    'home_lineup_stability', 'away_lineup_stability',
    'home_prior_red_cards', 'away_prior_red_cards',
    'home_card_rate_ewm', 'away_card_rate_ewm',
    'home_substitution_rate_ewm', 'away_substitution_rate_ewm',
    'home_lineup_known', 'away_lineup_known',
    'referee_home_win_rate', 'referee_draw_rate', 'referee_card_rate',
)


def from_record(row):
    return Match(str(row['season']), date.fromisoformat(str(row['date'])),
                 str(row['home']), str(row['away']),
                 int(row['hg']) if row.get('hg') not in (None, '') else None,
                 int(row['ag']) if row.get('ag') not in (None, '') else None,
                 int(row['matchday']) if row.get('matchday') not in (None, '') else None,
                 str(row.get('source_id', '')))


def _database_context_rows(connection, rows, target_season):
    """Derive pre-match lineup, discipline, substitution and referee features."""
    params = ('Premier League', 'England', target_season)
    match_ids = {m.source_id: m for m in rows}
    stats = {source_id: {'home_id': None, 'away_id': None, 'starters': {'home': set(), 'away': set()},
                         'cards': {'home': 0, 'away': 0}, 'red_cards': {'home': 0, 'away': 0},
                         'substitutions': {'home': 0, 'away': 0}, 'referees': set()}
             for source_id in match_ids}

    def fetch(query):
        cursor = connection.execute(query, params)
        columns = [column.name for column in cursor.description]
        return [dict(zip(columns, record)) for record in cursor.fetchall()]

    for record in fetch(DATABASE_TEAMS_SQL):
        item = stats.get(record['source_id'])
        if item is not None:
            item['home_id'], item['away_id'] = record['home_id'], record['away_id']
    for record in fetch(DATABASE_APPEARANCES_SQL):
        item = stats.get(record['source_id'])
        if item is None:
            continue
        item['home_id'], item['away_id'] = record['home_id'], record['away_id']
        side = 'home' if record['team_id'] == record['home_id'] else 'away'
        if record['started']:
            item['starters'][side].add(record['player_id'])
    for record in fetch(DATABASE_CARDS_SQL):
        item = stats.get(record['source_id'])
        if item is None or record['team_id'] is None:
            continue
        item['home_id'], item['away_id'] = record['home_id'], record['away_id']
        side = 'home' if record['team_id'] == record['home_id'] else 'away'
        item['cards'][side] += 1
        if record['card_type'] in ('red', 'yellow_red'):
            item['red_cards'][side] += 1
    for record in fetch(DATABASE_SUBSTITUTIONS_SQL):
        item = stats.get(record['source_id'])
        if item is None or record['team_id'] is None:
            continue
        item['home_id'], item['away_id'] = record['home_id'], record['away_id']
        side = 'home' if record['team_id'] == record['home_id'] else 'away'
        item['substitutions'][side] += 1
    for record in fetch(DATABASE_REFEREES_SQL):
        item = stats.get(record['source_id'])
        if item is not None:
            item['referees'].add(record['referee_id'])

    team_state = {}
    referee_state = {}
    context_rows = []
    last_year = None

    def team_record(team_id):
        return team_state.setdefault(team_id, {
            'last_starters': set(), 'previous_starters': set(), 'known': False,
            'card_rate': .0, 'substitution_rate': 0., 'last_red_cards': 0,
        })

    for match in rows:
        if match.year != last_year:
            # Starting lineups are not transferable evidence of the first
            # match of a new season; keep no stale XI across seasons.
            for record in team_state.values():
                record['last_starters'] = set()
                record['previous_starters'] = set()
                record['known'] = False
            last_year = match.year
        item = stats[match.source_id]
        home_id, away_id = item['home_id'], item['away_id']
        # For compact records without event rows, the team id is still known
        # from the match projection; use source names as a stable fallback.
        home_key, away_key = home_id or f'{match.home}:home', away_id or f'{match.away}:away'
        home = team_record(home_key)
        away = team_record(away_key)
        referee = next(iter(item['referees']), None)
        referee_stats = referee_state.get(referee, {'matches': 0, 'home_wins': 0, 'draws': 0, 'cards': 0})
        context_rows.append((fixture_key(match), (
            len(home['last_starters']), len(away['last_starters']),
            len(home['last_starters'] & home['previous_starters']) / max(len(home['last_starters']), 1),
            len(away['last_starters'] & away['previous_starters']) / max(len(away['last_starters']), 1),
            home['last_red_cards'], away['last_red_cards'],
            home['card_rate'], away['card_rate'],
            home['substitution_rate'], away['substitution_rate'],
            float(home['known']), float(away['known']),
            referee_stats['home_wins'] / max(referee_stats['matches'], 1),
            referee_stats['draws'] / max(referee_stats['matches'], 1),
            referee_stats['cards'] / max(referee_stats['matches'], 1),
        )))

        # Update only after the fixture's context has been recorded.
        for side, team_id in (('home', home_key), ('away', away_key)):
            record = team_state[team_id]
            current = item['starters'][side]
            record['previous_starters'] = record['last_starters']
            record['last_starters'] = current
            record['known'] = bool(current)
            record['card_rate'] = .9 * record['card_rate'] + .1 * item['cards'][side]
            record['substitution_rate'] = .9 * record['substitution_rate'] + .1 * item['substitutions'][side]
            record['last_red_cards'] = item['red_cards'][side]
        if referee is not None:
            referee_record = referee_state.setdefault(referee, {'matches': 0, 'home_wins': 0, 'draws': 0, 'cards': 0})
            referee_record['matches'] += 1
            referee_record['cards'] += item['cards']['home'] + item['cards']['away']
            if match.outcome == 0:
                referee_record['home_wins'] += 1
            elif match.outcome == 1:
                referee_record['draws'] += 1
    return MatchContext.from_rows(DATABASE_CONTEXT_FEATURES, context_rows)


def load_database_bundle(db_url=None, target_season='2026-27'):
    """One consistent read-only snapshot plus derived event context."""
    with connect(db_url) as conn:
        # connect() sets the timezone, beginning a transaction; end it first.
        conn.commit()
        conn.execute('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor = conn.execute(EXPORT_SQL, ('Premier League', 'England', target_season))
        columns = [c.name for c in cursor.description]
        rows = [from_record(dict(zip(columns, row))) for row in cursor.fetchall()]
        context = _database_context_rows(conn, rows, target_season)
    return rows, {'kind': 'supabase_postgres', 'read_only': True,
                  'database_context_features': list(context.feature_names)}, context


def load_database(db_url=None, target_season='2026-27'):
    """Backward-compatible match-only Supabase loader."""
    rows, provenance, _ = load_database_bundle(db_url, target_season)
    return rows, provenance


def load_csv(path):
    with Path(path).open(newline='') as f:
        rows = [from_record(row) for row in csv.DictReader(f)]
    return rows, {'kind': 'csv_snapshot', 'file': str(Path(path).resolve()),
                  'file_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def load_local(root, through='2025-26', alias_file='config/team_aliases.json'):
    aliases = load_alias_maps(alias_file)
    matches, stats, commit = load_scored_matches(Path(root), through=through, alias_maps=aliases)
    round_lookup = {}
    for source_file in sorted({m.source_file for m in matches}):
        parsed = parse_file(Path(root) / source_file, source_file=source_file)
        round_lookup.update({(source_file, m.source_match_index): m.matchday for m in parsed.matches})
    result = [Match(m.season_label, m.match_date, m.home_team, m.away_team,
                    m.home_goals, m.away_goals, matchday=round_lookup[(m.source_file, m.source_match_index)],
                    source_id=f'{m.source_file}:{m.source_match_index}')
              for m in matches if m.parse_status != 'failed']
    files = sorted({m.source_file for m in matches})
    return result, {'kind': 'local_openfootball_snapshot', 'repository_commit': commit,
                    'root': str(Path(root).resolve()), 'loader_stats': stats,
                    'files_sha256': {p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest()
                                     for p in files},
                    'supabase_snapshot_verified': False}


def load_fixtures(path, alias_file='config/team_aliases.json'):
    """Read scored AND unscored compact fixtures, intentionally discard scores.

    The ingestion result parser only reads scored matches. Using it here would
    silently simulate just ten matches of the supplied 2026–27 schedule.
    """
    text = Path(path).read_text()
    header, aliases = parse_header(text), load_alias_maps(alias_file)
    day, round_number, rows = None, None, []
    for index, line in enumerate(text.splitlines(), 1):
        round_match = re.match(r'^\s*▪\s*Matchday\s+(\d+)', line)
        if round_match:
            round_number = int(round_match[1])
            day = None
            continue
        parts = parse_date_parts(line, full_line=True)
        if parts:
            day = resolve_date(parts, header)
            continue
        if not re.search(r'\s+v\s+', line):
            continue
        _, without_time = parse_leading_time(line)
        score = SCORE_RE.search(without_time)
        if score:
            without_time = without_time[:score.start()]
        home, away = re.split(r'\s+v\s+', without_time.strip(), maxsplit=1)
        if day is None or round_number is None:
            raise ValueError(f'Fixture lacks date/matchday at {path}:{index}')
        rows.append(Match(header.season_label, day, aliases.target('team', home.strip()),
                          aliases.target('team', away.strip()), matchday=round_number,
                          source_id=f'{Path(path).name}:{index}'))
    validate_schedule(rows)
    return rows


def validate_schedule(rows, team_count=20):
    if not rows or len({m.season for m in rows}) != 1:
        raise ValueError('Schedule must contain exactly one season')
    teams = sorted({t for m in rows for t in (m.home, m.away)})
    pairs = Counter((m.home, m.away) for m in rows)
    expected = {(h, a) for h in teams for a in teams if h != a}
    if len(teams) != team_count or set(pairs) != expected or any(v != 1 for v in pairs.values()):
        raise ValueError(f'Incomplete/duplicate schedule: {len(rows)} matches, {len(teams)} teams; '
                         f'expected {team_count * (team_count - 1)} matches')
    appearances = Counter((m.date, t) for m in rows for t in (m.home, m.away))
    if any(v > 1 for v in appearances.values()):
        raise ValueError('A club has multiple matches on the same date')
    if all(m.matchday is not None for m in rows):
        rounds = Counter(m.matchday for m in rows)
        if set(rounds) != set(range(1, 2 * (team_count - 1) + 1)) or any(v != team_count // 2 for v in rounds.values()):
            raise ValueError('Matchdays do not form a complete season')
    return teams


def audit_history(rows, target_season):
    cutoff = int(target_season[:4])
    history = [m for m in rows if m.year < cutoff]
    if any(m.date is None for m in history):
        raise ValueError('Historical matches require known dates')
    history.sort(key=lambda m: (m.date, m.home, m.away))
    if not history:
        raise ValueError('No historical matches before target season')
    seen = set()
    for m in history:
        if m.date is None or m.hg is None or m.ag is None or min(m.hg, m.ag) < 0:
            raise ValueError(f'Invalid completed match: {m.source_id}')
        if not (date(m.year, 7, 1) <= m.date < date(m.year + 1, 7, 1)):
            # COVID season legitimately ran until July 2020.
            if not (m.season == '2019-20' and m.date < date(2020, 8, 1)):
                raise ValueError(f'Date outside season: {m.source_id}')
        key = (m.season, m.home, m.away)
        if key in seen or m.home == m.away:
            raise ValueError(f'Duplicate/self fixture: {key}')
        seen.add(key)
    coverage = {}
    for season in sorted({m.season for m in history}):
        subset = [m for m in history if m.season == season]
        validate_schedule(subset)
        coverage[season] = len(subset)
    return history, {'matches': len(history), 'coverage': coverage,
                     'excluded_target_or_later_rows': sum(m.year >= cutoff for m in rows),
                     'latest_training_match': str(max(m.date for m in history))}


def write_matches(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(Match.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(m) for m in rows)


def fingerprint(rows):
    return hashlib.sha256(json.dumps([asdict(m) for m in rows], default=str, sort_keys=True).encode()).hexdigest()


def points_table(matches):
    totals = {}
    for m in matches:
        for t, gf, ga in ((m.home, m.hg, m.ag), (m.away, m.ag, m.hg)):
            record = totals.setdefault(t, [0, 0, 0])
            record[0] += 3 if gf > ga else 1 if gf == ga else 0
            record[1] += gf - ga
            record[2] += gf
    return sorted(totals, key=lambda t: (*(-v for v in totals[t]), t))


def infer_participants(history, championship_path, alias_file='config/team_aliases.json'):
    """Infer membership from prior-season sporting results, not future files.

    Points deductions and administrative changes are outside the fact schema.
    Membership is therefore recorded as a source-derived scenario.
    """
    previous = max(m.season for m in history)
    pl = [m for m in history if m.season == previous]
    relegated = points_table(pl)[-3:]
    parsed = parse_file(Path(championship_path))
    if parsed.season_directory != previous:
        raise ValueError('Championship source must be the last completed PL season')
    aliases = load_alias_maps(alias_file)
    regular, playoffs = [], []
    for m in parsed.matches:
        if m.score is None or m.home_source_name is None or m.away_source_name is None or m.match_date is None:
            raise ValueError('Incomplete Championship result')
        row = Match(previous, m.match_date, aliases.target('team', m.home_source_name),
                    aliases.target('team', m.away_source_name), m.score.home_score, m.score.away_score)
        (playoffs if m.round_info and 'playoff' in m.round_info.round_name.lower() else regular).append(row)
    validate_schedule(regular, team_count=24)
    if len(playoffs) != 5:
        raise ValueError('Expected five Championship playoff matches')
    final = max(playoffs, key=lambda m: m.date)
    if final.hg == final.ag:
        raise ValueError('Playoff final winner requires an explicit penalties result')
    winner = final.home if final.hg > final.ag else final.away
    promoted = points_table(regular)[:2] + [winner]
    participants = sorted((set(points_table(pl)) - set(relegated)) | set(promoted))
    if len(participants) != 20 or len(set(promoted)) != 3:
        raise ValueError('Inferred membership does not contain 20 unique clubs')
    warnings = []
    finalists = {final.home, final.away}
    semifinals = [m for m in playoffs if m is not final]
    for team in finalists:
        games = [m for m in semifinals if team in (m.home, m.away)]
        aggregate = sum(m.hg - m.ag if m.home == team else m.ag - m.hg for m in games)
        if len(games) != 2 or aggregate < 0:
            warnings.append(f'{team} appears in the recorded final despite an inconsistent semifinal record; membership uses the recorded final winner.')
    return participants, {'method': 'Prior-season PL bottom three, Championship top two and recorded playoff-final winner; match-derived points without sanctions.',
                           'promoted': promoted, 'relegated': relegated,
                           'championship_file_sha256': hashlib.sha256(Path(championship_path).read_bytes()).hexdigest(),
                           'source_warnings': warnings, 'official_membership_verified': False}


def generated_schedule(teams, season, seed=202627):
    """Seeded circle-method double round robin. Dates are illustrative only."""
    import random
    rotating = list(sorted(teams))
    if len(rotating) % 2 or len(set(rotating)) != len(rotating):
        raise ValueError('Schedule requires an even number of unique clubs')
    random.Random(seed).shuffle(rotating)
    first_leg = []
    for day in range(len(teams) - 1):
        pairs = []
        for i in range(len(teams) // 2):
            h, a = rotating[i], rotating[-1 - i]
            if (day + i) % 2:
                h, a = a, h
            pairs.append((h, a))
        first_leg.append(pairs)
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    rounds = first_leg + [[(a, h) for h, a in pairs] for pairs in first_leg]
    start = date(int(season[:4]), 8, 15)
    start += timedelta(days=(5 - start.weekday()) % 7)
    fixtures = [Match(season, start + timedelta(days=7 * day), h, a, matchday=day + 1,
                      source_id=f'generated:{seed}:{day + 1}:{h}:{a}')
                for day, pairs in enumerate(rounds) for h, a in pairs]
    validate_schedule(fixtures, team_count=len(teams))
    return fixtures
