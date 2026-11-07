# OpenFootball format notes

These notes were produced by inspecting the current `openfootball/england` clone at implementation time.

Inspection snapshot: commit `836b1947fa4089c86b6064f821eee7de926a7a3f`.

## Observed source layouts

- `2000-01`, `2010-11`, and `2019-20` contain `1-premierleague.txt`. They use compact day blocks: a date line such as `Sat Aug 19 2000` (or `Sat Aug 17`), a kickoff time on the first match of a time group, and subsequent results with the time omitted. Team names and scores are on the same line.
- `2025-26` contains both files. `1-premierleague-full.txt` uses one match metadata line such as `Sat Aug 16 12:30 UTC+1 @ Villa Park, Birmingham, Att: 42526`, followed by a `home v away score` line, a scorer block, team lineups, cards, and `Refs:`. The compact file also has scorer blocks but no rich metadata.
- The repository snapshot inspected also contains `2026-27/1-premierleague.txt`; it is a compact/in-progress season file and is handled by the same fallback logic.
- Full-file round headings are `▪ Regular Season - N`; compact historical headings are `▪ Matchday N`. The parser stores both the parsed number and exact heading.
- Some compact historical score lines omit halftime scores, e.g. `0-0`; halftime fields stay `NULL` in those records.
- Full files contain both `UTC+0` and `UTC+1` on the same matchday. The importer uses the source offset on each metadata line instead of deriving it from the month.

## Event edge cases observed

- Goals use semicolon-separated home/away groups, repeated minutes belong to the preceding scorer (`Chris WOOD 5', 45+2'`), and annotations include `(p)` and `(og)`.
- Lineups use commas and visual `-` group separators. The parser preserves group indexes but does not label positions. Nested substitutions occur, for example `A (16' B (75' C))`, meaning two linked substitution events.
- Captains are marked with `[c]`; the marker is removed only for identity comparison and remains in `raw_player_text`/`raw_match_text`.
- Card lines are independent `Yellow:` and `Red:` lines. Direct red versus second yellow is not inferred.
- Referees are comma-separated after `Refs:` and are stored in source order with a `NULL` role.
- The inspected rich snapshot also contains explicit `Yellow-Red:` card lines. These are stored as `yellow_red`, preserving the source distinction without inferring whether the dismissal was a direct red or a second yellow.

The parser records warnings for malformed or unfamiliar fragments rather than dropping the enclosing match. A season can therefore be imported and audited even when one record is imperfect.
