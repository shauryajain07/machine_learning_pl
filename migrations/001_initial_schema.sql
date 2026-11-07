-- Premier League historical data schema for Supabase/PostgreSQL.
-- Apply this file once from the Supabase SQL editor or with:
--   python -m src.scripts.migrate_supabase
--
-- This migration deliberately contains no standings, ratings, Elo, or ML
-- feature tables. It stores source facts and parser/audit metadata only.

create extension if not exists pgcrypto;

create table if not exists schema_migrations (
    version text primary key,
    applied_at timestamptz not null default now()
);

create table if not exists competitions (
    competition_id bigint generated always as identity primary key,
    name text not null,
    country text not null,
    created_at timestamptz not null default now(),
    unique (name, country)
);

create table if not exists seasons (
    season_id bigint generated always as identity primary key,
    competition_id bigint not null references competitions(competition_id),
    season_label text not null,
    start_date date,
    end_date date,
    expected_team_count integer,
    expected_match_count integer,
    source_header_text text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (competition_id, season_label),
    check (expected_team_count is null or expected_team_count >= 0),
    check (expected_match_count is null or expected_match_count >= 0)
);

create table if not exists rounds (
    round_id bigint generated always as identity primary key,
    season_id bigint not null references seasons(season_id) on delete cascade,
    round_name text not null,
    round_number integer,
    raw_heading text not null,
    unique (season_id, raw_heading)
);

create table if not exists teams (
    team_id bigint generated always as identity primary key,
    canonical_name text not null,
    normalized_name text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists team_aliases (
    alias_id bigint generated always as identity primary key,
    team_id bigint not null references teams(team_id) on delete cascade,
    alias text not null,
    normalized_alias text not null,
    source text not null,
    unique (team_id, normalized_alias, source)
);

create table if not exists players (
    player_id bigint generated always as identity primary key,
    canonical_name text not null,
    normalized_name text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists player_aliases (
    alias_id bigint generated always as identity primary key,
    player_id bigint not null references players(player_id) on delete cascade,
    alias text not null,
    normalized_alias text not null,
    source text not null,
    unique (player_id, normalized_alias, source)
);

create table if not exists referees (
    referee_id bigint generated always as identity primary key,
    canonical_name text not null,
    normalized_name text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists ingestion_runs (
    ingestion_run_id bigint generated always as identity primary key,
    source_repository text not null,
    source_root text not null,
    source_commit_sha text,
    options jsonb not null default '{}'::jsonb,
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    status text not null default 'running'
        check (status in ('running', 'completed', 'failed', 'dry_run')),
    counts jsonb not null default '{}'::jsonb,
    error_text text
);

create table if not exists source_files (
    source_file_id bigint generated always as identity primary key,
    ingestion_run_id bigint references ingestion_runs(ingestion_run_id),
    source_repository text not null,
    source_root text not null,
    season_directory text not null,
    source_filename text not null,
    source_path text not null,
    source_commit_sha text,
    file_sha256 text not null,
    file_format text not null check (file_format in ('full', 'compact')),
    file_size_bytes bigint not null check (file_size_bytes >= 0),
    header_text text not null default '',
    ingested_at timestamptz not null default now(),
    unique (source_repository, source_path, file_sha256)
);

create table if not exists matches (
    match_id bigint generated always as identity primary key,
    season_id bigint not null references seasons(season_id),
    round_id bigint references rounds(round_id),
    matchday integer,
    match_date date,
    source_weekday text,
    -- ISO weekday: Monday=1 through Sunday=7. It is stored for simple,
    -- indexable queries and is checked against match_date when both exist.
    day_of_week smallint,
    kickoff_local_time time,
    utc_offset_minutes smallint,
    kickoff_utc timestamptz,
    venue text,
    location text,
    attendance integer check (attendance is null or attendance >= 0),
    home_team_id bigint references teams(team_id),
    away_team_id bigint references teams(team_id),
    home_score smallint check (home_score is null or home_score >= 0),
    away_score smallint check (away_score is null or away_score >= 0),
    home_ht_score smallint check (home_ht_score is null or home_ht_score >= 0),
    away_ht_score smallint check (away_ht_score is null or away_ht_score >= 0),
    source_file_id bigint references source_files(source_file_id),
    source_file text not null,
    source_match_index integer not null check (source_match_index > 0),
    source_fingerprint text not null,
    source_format text not null check (source_format in ('full', 'compact')),
    raw_match_text text not null,
    raw_goal_block text,
    raw_home_lineup_text text,
    raw_away_lineup_text text,
    parse_status text not null check (parse_status in ('clean', 'warning', 'failed')),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (
        home_team_id is null or away_team_id is null or home_team_id <> away_team_id
    ),
    check (day_of_week is null or day_of_week between 1 and 7),
    check (match_date is null or day_of_week is null or extract(isodow from match_date) = day_of_week),
    unique (season_id, source_file, source_match_index),
    unique (season_id, source_fingerprint)
);

create table if not exists season_teams (
    season_id bigint not null references seasons(season_id) on delete cascade,
    team_id bigint not null references teams(team_id),
    first_source_name text,
    source_names jsonb not null default '[]'::jsonb,
    primary key (season_id, team_id)
);

create table if not exists match_team_names (
    match_id bigint not null references matches(match_id) on delete cascade,
    team_id bigint references teams(team_id),
    side text not null check (side in ('home', 'away')),
    source_name text not null,
    primary key (match_id, side)
);

create table if not exists appearances (
    appearance_id bigint generated always as identity primary key,
    match_id bigint not null references matches(match_id) on delete cascade,
    team_id bigint not null references teams(team_id),
    player_id bigint not null references players(player_id),
    started boolean not null,
    captain boolean not null default false,
    entered_minute smallint,
    entered_stoppage_time smallint,
    entered_absolute_minute smallint,
    exited_minute smallint,
    exited_stoppage_time smallint,
    exited_absolute_minute smallint,
    source_order integer not null,
    raw_group_index integer,
    raw_player_text text not null,
    unique (match_id, team_id, player_id),
    check (entered_minute is null or entered_minute >= 0),
    check (exited_minute is null or exited_minute >= 0)
);

create table if not exists substitutions (
    substitution_id bigint generated always as identity primary key,
    match_id bigint not null references matches(match_id) on delete cascade,
    team_id bigint references teams(team_id),
    player_off_id bigint references players(player_id),
    player_on_id bigint references players(player_id),
    minute smallint,
    stoppage_time smallint,
    absolute_minute smallint,
    raw_minute text,
    source_order integer not null,
    raw_event_text text not null,
    check (player_off_id is null or player_on_id is null or player_off_id <> player_on_id),
    check (minute is null or minute >= 0),
    check (stoppage_time is null or stoppage_time >= 0),
    unique (match_id, source_order)
);

create table if not exists goals (
    goal_id bigint generated always as identity primary key,
    match_id bigint not null references matches(match_id) on delete cascade,
    scoring_team_id bigint not null references teams(team_id),
    player_id bigint references players(player_id),
    minute smallint,
    stoppage_time smallint,
    absolute_minute smallint,
    penalty boolean not null default false,
    own_goal boolean not null default false,
    goal_order integer not null,
    raw_minute text,
    annotation text,
    raw_event_text text not null,
    unique (match_id, goal_order),
    check (minute is null or minute >= 0),
    check (stoppage_time is null or stoppage_time >= 0)
);

create table if not exists cards (
    card_id bigint generated always as identity primary key,
    match_id bigint not null references matches(match_id) on delete cascade,
    team_id bigint references teams(team_id),
    player_id bigint references players(player_id),
    -- Yellow-Red is an explicit source label and is retained separately;
    -- it is not inferred as a direct red or a second yellow.
    card_type text not null check (card_type in ('yellow', 'red', 'yellow_red')),
    minute smallint,
    stoppage_time smallint,
    absolute_minute smallint,
    raw_minute text,
    raw_event_text text not null,
    source_order integer not null,
    unique (match_id, card_type, source_order),
    check (minute is null or minute >= 0),
    check (stoppage_time is null or stoppage_time >= 0)
);

create table if not exists match_referees (
    match_id bigint not null references matches(match_id) on delete cascade,
    referee_id bigint not null references referees(referee_id),
    referee_order integer not null,
    role text,
    primary key (match_id, referee_order)
);

create table if not exists parsing_issues (
    issue_id bigint generated always as identity primary key,
    ingestion_run_id bigint references ingestion_runs(ingestion_run_id),
    match_id bigint references matches(match_id) on delete cascade,
    source_file text not null,
    source_match_index integer,
    severity text not null check (severity in ('info', 'warning', 'error')),
    category text not null,
    raw_fragment text,
    description text not null,
    created_at timestamptz not null default now()
);

create table if not exists identity_review_candidates (
    candidate_id bigint generated always as identity primary key,
    entity_type text not null check (entity_type in ('team', 'player', 'referee')),
    source_value text not null,
    normalized_value text not null,
    candidate_entity_id bigint,
    source_file text,
    evidence text not null,
    status text not null default 'pending'
        check (status in ('pending', 'accepted', 'rejected', 'reviewed')),
    created_at timestamptz not null default now(),
    unique (entity_type, source_value, normalized_value, candidate_entity_id, status)
);

create index if not exists idx_matches_season_kickoff
    on matches (season_id, kickoff_utc);
create index if not exists idx_matches_home_kickoff
    on matches (home_team_id, kickoff_utc);
create index if not exists idx_matches_away_kickoff
    on matches (away_team_id, kickoff_utc);
create index if not exists idx_matches_match_date
    on matches (match_date, day_of_week, matchday);
create index if not exists idx_appearances_match_team
    on appearances (match_id, team_id);
create index if not exists idx_appearances_player
    on appearances (player_id);
create index if not exists idx_appearances_team
    on appearances (team_id);
create index if not exists idx_goals_match
    on goals (match_id);
create index if not exists idx_goals_player
    on goals (player_id);
create index if not exists idx_goals_scoring_team
    on goals (scoring_team_id);
create index if not exists idx_goals_absolute_minute
    on goals (absolute_minute);
create index if not exists idx_substitutions_match
    on substitutions (match_id);
create index if not exists idx_substitutions_team
    on substitutions (team_id);
create index if not exists idx_cards_match
    on cards (match_id);
create index if not exists idx_cards_player
    on cards (player_id);
create index if not exists idx_cards_team
    on cards (team_id);
create index if not exists idx_match_referees_match
    on match_referees (match_id);
create index if not exists idx_match_referees_referee
    on match_referees (referee_id);
create index if not exists idx_parsing_issues_match
    on parsing_issues (match_id, severity);
create index if not exists idx_identity_candidates_status
    on identity_review_candidates (status, entity_type);

create or replace view season_coverage as
select
    s.season_id,
    c.name as competition_name,
    s.season_label,
    coalesce(bool_or(sf.file_format = 'full'), false) as full_file_available,
    count(distinct m.match_id) as matches,
    count(distinct m.match_id) filter (
        where exists (select 1 from appearances a where a.match_id = m.match_id)
    ) as lineups,
    count(distinct sub.substitution_id) as substitutions,
    count(distinct g.goal_id) as goals,
    count(distinct card.card_id) filter (where card.card_type = 'yellow') as yellow_cards,
    count(distinct card.card_id) filter (where card.card_type = 'red') as red_cards,
    count(distinct card.card_id) filter (where card.card_type = 'yellow_red') as yellow_red_cards,
    count(distinct mr.referee_id) as referees,
    count(distinct m.match_id) filter (where m.attendance is not null) as attendance
from seasons s
join competitions c on c.competition_id = s.competition_id
left join matches m on m.season_id = s.season_id
left join source_files sf on sf.source_file_id = m.source_file_id
left join substitutions sub on sub.match_id = m.match_id
left join goals g on g.match_id = m.match_id
left join cards card on card.match_id = m.match_id
left join match_referees mr on mr.match_id = m.match_id
group by s.season_id, c.name, s.season_label;

comment on table matches is 'One source match record; raw_match_text is the exact OpenFootball source block.';
comment on column matches.matchday is 'Source round number/matchday when present; no standings are calculated here.';
comment on column goals.absolute_minute is 'minute + stoppage_time; raw_minute remains authoritative for source semantics.';
comment on column goals.goal_order is 'Order in the source scorer block (home group then away group), not an inferred chronological order.';
comment on column matches.day_of_week is 'ISO weekday of the local match date, Monday=1 through Sunday=7.';
