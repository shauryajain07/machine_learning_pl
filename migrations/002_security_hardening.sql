-- Keep the source database private-by-default while retaining authenticated
-- database-admin access for the importer and Supabase MCP.

alter view if exists public.season_coverage
    set (security_invoker = true);

-- This pre-existing event-trigger helper must remain executable by PostgreSQL
-- internally, but it is not an application RPC endpoint.
revoke execute on function public.rls_auto_enable() from public, anon, authenticated;

revoke all on table public.schema_migrations from anon, authenticated;

revoke insert, update, delete, truncate, references, trigger
    on table public.competitions,
              public.seasons,
              public.rounds,
              public.teams,
              public.team_aliases,
              public.players,
              public.player_aliases,
              public.referees,
              public.ingestion_runs,
              public.source_files,
              public.matches,
              public.season_teams,
              public.match_team_names,
              public.appearances,
              public.substitutions,
              public.goals,
              public.cards,
              public.match_referees,
              public.parsing_issues,
              public.identity_review_candidates
    from anon, authenticated;
