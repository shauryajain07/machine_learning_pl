-- Cover foreign-key lookups and parent updates without changing the data model.

create index if not exists idx_matches_round
    on public.matches (round_id);
create index if not exists idx_matches_source_file
    on public.matches (source_file_id);
create index if not exists idx_source_files_ingestion_run
    on public.source_files (ingestion_run_id);
create index if not exists idx_season_teams_team
    on public.season_teams (team_id);
create index if not exists idx_match_team_names_team
    on public.match_team_names (team_id);
create index if not exists idx_substitutions_player_off
    on public.substitutions (player_off_id);
create index if not exists idx_substitutions_player_on
    on public.substitutions (player_on_id);
create index if not exists idx_parsing_issues_ingestion_run
    on public.parsing_issues (ingestion_run_id);
