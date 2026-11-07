"""Portable, offline results report with a matchday explorer."""
import html
import json


def write_report(output, report, simulation):
    selected = report['tuning']['selected']
    source = report['source']
    membership = source['fixture_source'].get('membership', {})
    text = [f'# Premier League {simulation["season"]} simulation', '',
            f'{simulation["n_simulations"]:,} simulated seasons; seed {simulation["seed"]}.', '',
            f'Source: **{source["kind"]}**. Training ends {source["audit"]["latest_training_match"]}. '
            f'{source["audit"]["matches"]:,} historical matches. Live Supabase snapshot verification: '
            f'{source.get("supabase_snapshot_verified", source["kind"] == "supabase_postgres")}.', '',
            'The delivered run ignores all 2026–27 source files. The schedule is a generated '
            '38-matchday double round robin with illustrative dates. Club membership is inferred '
            'from prior-season match results; administrative points deductions are not represented.', '',
            '## Selected model', '',
            f'- Goal model: `{selected["goals"]["family"]}`; recency half-life {selected["goals"]["half_life"]} days.',
            f'- Outcome model: `{selected["outcomes"]["family"]}`; blend weight {selected["blend"]:.0%}.',
            f'- Dixon–Coles low-score correction rho: {selected["rho"]}.',
            '- Training weight: 2 ** (−age_days / half_life_days), normalized to mean one.',
            '- More recent games also drive Elo and exponentially weighted scoring/conceding form.', '',
            'Model selection used annual expanding-window validation on 2021–22, 2022–23 and 2023–24. '
            'The search compared regularized Poisson regression, Poisson boosted trees, logistic regression '
            'and outcome boosted trees. The score distribution is the basis for match scores and league standings. '
            'All models are fitted with recency weights. Trees use fixed iteration budgets, with random early-stopping splits disabled.', '',
            '## Reserved chronological evaluation', '',
            '| Season | Selected log loss | Historical prior | Elo | Accuracy | Score NLL |',
            '|---|---:|---:|---:|---:|---:|']
    for year, values in report['test'].items():
        m = values['selected']
        text.append(f'| {year}–{str(int(year)+1)[-2:]} | {m["log_loss"]:.4f} | '
                    f'{values["historical_outcome_prior"]["log_loss"]:.4f} | '
                    f'{values["fixed_elo_baseline"]["log_loss"]:.4f} | {m["accuracy"]:.1%} | {m["score_nll"]:.4f} |')
    text += ['', 'Lower log loss and score NLL are better. Features update after each completed date. '
             'These are historical replay metrics, not promises of future performance. The project previously '
             'evaluated older baselines on these seasons; this new pipeline reserves them from its tuning. '
             'The 2025–26 refit includes 2024–25, as it would in deployment.', '',
             '## Full-season forecast backtests', '',
             '| Season | Points MAE | Coverage of 90% points interval | Simulated seasons |',
             '|---|---:|---:|---:|']
    for year, values in report['preseason_backtests'].items():
        text.append(f'| {year} | {values["points_mae"]:.2f} | {values["points_90pct_interval_coverage"]:.0%} | {values["n_simulations"]:,} |')
    text += ['', 'Each preseason backtest freezes training before the season and recursively simulates all '
             '380 results. No actual results from that season enter those simulations. Target scores and '
             'season features are never used before they occur.', '', '## Season forecast', '',
             '| Club | Expected points | 90% interval | Title | Top four | Relegation |',
             '|---|---:|---:|---:|---:|---:|']
    for r in simulation['table']:
        text.append(f'| {r["team"]} | {r["expected_points"]:.1f} | {r["points_p05"]:.0f}–{r["points_p95"]:.0f} | '
                    f'{r["title_probability"]:.1%} | {r["top4_probability"]:.1%} | {r["relegation_probability"]:.1%} |')
    text += ['', '## Assumptions and limits', '',
             '- Probabilities condition on this dataset, generated schedule, inferred membership and fitted model.',
             '- Intervals include random match outcomes and recursive form changes; they exclude model-parameter uncertainty, transfers, injuries, lineups, managerial changes, and sanctions.',
             '- Promoted clubs receive a conservative shared prior. Championship results determine membership only, not cross-division strength calibration.',
             '- Top four means finishing positions 1–4; it is not a forecast of UEFA qualification rules.',
             '- Ties use points, goal difference, goals for, head-to-head points and away goals. Remaining ties use a seeded random playoff approximation.',
             '- Goal counts are bounded at 20 for computation; goal-rate bounds are 0.08–5.5. The omitted Poisson tail at the largest rate is below 4e-7.',
             '- Reported fixture probabilities average over earlier simulated results. Replaying with a different seed changes Monte Carlo draws, not model training.',
             '- Refresh source data and rerun training before relying on the forecast as current.', '',
             f'Promoted clubs in this scenario: {", ".join(membership.get("promoted", simulation["promoted_teams"]))}.',
             f'Relegated clubs from prior-season results: {", ".join(membership.get("relegated", []))}.', '']
    for warning in membership.get('source_warnings', []):
        text += [f'**Source consistency note:** {warning}', '']
    text += ['## Reproduce', '', '```bash',
             ".venv/bin/python -m src.scripts.train_season_simulator --source local",
             '.venv/bin/python -m src.scripts.simulate_season --seed 202627', '```', '',
             'For live database training, set `SUPABASE_DB_URL` in the environment and run with `--source supabase`. '
             'The database loader runs a consistent read-only transaction. It has not been integration-tested against '
             'the live project in this session. A connection failure never silently falls back to local data.', '',
             'Artifacts: `model.joblib`, `metrics.json`, `data_audit.json`, `historical_matches.csv`, '
             '`training_features.csv`, `test_predictions.csv`, `target_fixtures.csv`, '
             '`fixture_predictions.csv`, `season_forecast.csv`, `simulation.json`, `simulated_scores.npz`.', '',
             '## Method references', '',
             '- [Dixon & Coles (1997), original football score model](https://www.research.lancs.ac.uk/portal/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-football-betting-market%28d16276a2-d6e0-483b-a708-1d29663f1992%29.html)',
             '- [scikit-learn: histogram boosting, Poisson loss and sample weights](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)',
             '- [Premier League table tie-break explanation](https://www.premierleague.com/en/premier-league-explained)', '']
    (output / 'REPORT.md').write_text('\n'.join(text))
    data = json.dumps({'simulation': simulation, 'metrics': report['test'], 'model': selected,
                       'source': source['kind'], 'backtests': report['preseason_backtests']}).replace('<', '\\u003c')
    title = html.escape(f'Premier League {simulation["season"]}')
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Season simulator</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0b1220;color:#eaf0f7}body{max-width:1160px;margin:auto;padding:42px 24px}h1{font-size:42px;letter-spacing:-1.5px;margin:12px 0}h2{font-size:22px;margin-top:36px}p{max-width:900px;line-height:1.6;color:#b7c5d6}a{color:#60dfb5}.tag{color:#60dfb5;font-size:13px;letter-spacing:2px;text-transform:uppercase}.cards{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0}.card{background:#142137;border:1px solid #273953;border-radius:12px;padding:20px;flex:1;min-width:180px}.card strong{display:block;font-size:29px;color:#fff}.card span{font-size:13px;color:#a7b9cc}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:#95acc7;font-weight:500;white-space:nowrap}td,th{padding:12px 10px;border-bottom:1px solid #243249}td.num{text-align:right;font-variant-numeric:tabular-nums}th.num{text-align:right}tbody tr:hover{background:#132139}input[type=range]{width:min(550px,90%);accent-color:#60dfb5}select{padding:8px;background:#18283e;color:white;border:1px solid #36506d;border-radius:6px}.note{border-left:3px solid #e2b66f;padding-left:15px}button{background:#234638;color:#b9f8de;border:0;border-radius:6px;padding:10px;cursor:pointer}.fixture-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}.fixture{padding:14px;background:#142137;border-radius:8px}.fixture small{color:#a1b3c9}.footer{font-size:13px;margin-top:40px}
</style>
<div class="tag">Historical learning / Monte Carlo simulation</div><h1>__TITLE__</h1>
<p>Explore 38 matchdays across 10,000 possible seasons. Recent seasons carry more weight. Every simulated result updates each club’s form before its next match.</p>
<p class="note">Scenario schedule: generated fixtures and illustrative dates. All 2026–27 source files are excluded. Membership comes from prior-season results. This run uses a pinned local source snapshot; live Supabase contents were not verified.</p>
<div class="cards" id="cards"></div>
<h2>Season probabilities</h2><p>Clubs ordered by expected finishing position. Points ranges contain the middle 90% of simulations; they do not include every source of uncertainty.</p>
<div class="scroll"><table><thead><tr><th>Club</th><th class="num">Points</th><th class="num">90% range</th><th class="num">Title</th><th class="num">Top 4</th><th class="num">Relegation</th></tr></thead><tbody id="forecast"></tbody></table></div>
<h2>Matchday explorer · <span id="dayLabel">1</span></h2><p><label for="day">Matchday</label><br><input id="day" type="range" min="1" max="38" value="1"></p>
<label for="mode">Table view </label><select id="mode"><option value="expected">Average across seasons</option><option value="sample">One simulated season</option></select>
<div class="scroll"><table><thead id="dayHead"></thead><tbody id="dayTable"></tbody></table></div><h2>Matches</h2><div id="matches" class="fixture-grid"></div>
<h2>Historical evaluation</h2><div class="scroll"><table><thead><tr><th>Season</th><th>Model log loss</th><th>Historical prior</th><th>Elo baseline</th><th>Accuracy</th></tr></thead><tbody id="evaluation"></tbody></table></div>
<p>Lower log loss is better. Models were selected using 2021–24 validation; these evaluation seasons did not select hyperparameters. Full-season backtests and calibration bins are in the detailed report.</p>
<p class="note">The model uses historical scores and form. It does not know transfers, injuries, lineups or future sanctions. Promoted clubs use a shared conservative prior. The supplied Championship playoff record contains an inconsistency; the membership scenario uses the recorded final winner. Top four describes league positions, not European qualification.</p>
<p class="footer"><a href="REPORT.md">Detailed methodology and results</a> · <a href="season_forecast.csv">Download forecast CSV</a> · <a href="fixture_predictions.csv">Download match CSV</a> · <a href="metrics.json">Model diagnostics</a></p>
<script id="payload" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('payload').textContent),s=d.simulation, $=id=>document.getElementById(id), pct=x=>(x*100).toFixed(1)+'%', esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('cards').innerHTML=`<div class="card"><strong>${s.n_simulations.toLocaleString()}</strong><span>Simulated seasons · seed ${s.seed}</span></div><div class="card"><strong>${d.model.goals.half_life} days</strong><span>Training-weight half-life</span></div><div class="card"><strong>${s.table[0].team}</strong><span>Highest expected finish · ${pct(s.table[0].title_probability)} title probability</span></div>`;
$('forecast').innerHTML=s.table.map(r=>`<tr><td>${esc(r.team)}</td><td class="num">${r.expected_points.toFixed(1)}</td><td class="num">${r.points_p05.toFixed(0)}–${r.points_p95.toFixed(0)}</td><td class="num">${pct(r.title_probability)}</td><td class="num">${pct(r.top4_probability)}</td><td class="num">${pct(r.relegation_probability)}</td></tr>`).join('');
$('evaluation').innerHTML=Object.entries(d.metrics).map(([year,r])=>`<tr><td>${year}–${String(+year+1).slice(-2)}</td><td>${r.selected.log_loss.toFixed(4)}</td><td>${r.historical_outcome_prior.log_loss.toFixed(4)}</td><td>${r.fixed_elo_baseline.log_loss.toFixed(4)}</td><td>${pct(r.selected.accuracy)}</td></tr>`).join('');
function render(){const day=+$('day').value, sample=$('mode').value==='sample';$('dayLabel').textContent=day;let rows=[...s.matchdays.find(x=>x.matchday===day).table];
if(sample)rows.sort((a,b)=>b.sample_points-a.sample_points||(b.sample_gf-b.sample_ga)-(a.sample_gf-a.sample_ga)||b.sample_gf-a.sample_gf||a.team.localeCompare(b.team));
$('dayHead').innerHTML=sample?'<tr><th>Club</th><th>Points</th><th>GF</th><th>GA</th></tr>':'<tr><th>Club</th><th>Expected points</th><th>90% range</th></tr>';
$('dayTable').innerHTML=rows.map(r=>sample?`<tr><td>${esc(r.team)}</td><td>${r.sample_points}</td><td>${r.sample_gf}</td><td>${r.sample_ga}</td></tr>`:`<tr><td>${esc(r.team)}</td><td>${r.expected_points.toFixed(1)}</td><td>${r.p05.toFixed(0)}–${r.p95.toFixed(0)}</td></tr>`).join('');
$('matches').innerHTML=s.fixtures.filter(x=>x.matchday===day).map(f=>`<div class="fixture">${esc(f.home)} <b>${sample?f.sample_home_goals+' – '+f.sample_away_goals:'vs'}</b> ${esc(f.away)}<br><small>Home ${pct(f.home_win)} · Draw ${pct(f.draw)} · Away ${pct(f.away_win)}</small></div>`).join('');}
$('day').addEventListener('input',render);$('mode').addEventListener('change',render);render();
</script></html>"""
    (output / 'report.html').write_text(page.replace('__TITLE__', title).replace('__DATA__', data))
