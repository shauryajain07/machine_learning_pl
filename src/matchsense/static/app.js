"use strict";
const $ = (sel) => document.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
let health = null;
let lastQuestion = null;
let autoTimer = null;

function tz() { return $("#tz").value; }
function fmtTime(iso) {
  if (!iso) return "—";
  return new Intl.DateTimeFormat("en-GB", { timeZone: tz(), weekday: "short", day: "2-digit", month: "short",
    hour: "2-digit", minute: "2-digit" }).format(new Date(iso));
}
function ago(iso, nowIso) {
  const ms = new Date(nowIso || Date.now()) - new Date(iso);
  const h = ms / 36e5;
  if (h < 1) return `${Math.max(1, Math.round(h * 60))} min ago`;
  if (h < 48) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

async function loadHealth() {
  health = await fetch("/api/health").then((r) => r.json());
  const badge = $("#modeBadge");
  badge.textContent = health.mode === "demo" ? "Demo · recorded + simulated" : "Live";
  badge.className = "badge" + (health.mode === "demo" ? " demo" : "");
  try { const saved = localStorage.getItem("ms-tz"); if (saved) $("#tz").value = saved; else $("#tz").value = health.timezone; } catch (_) { $("#tz").value = health.timezone; }
  if (health.mode === "demo" && health.demo) {
    $("#demoBanner").hidden = false;
    $("#demoBanner").textContent = `DEMO MODE — FPL data recorded ${health.demo.fpl_recorded_at}; news and official lineups are SIMULATED (fictional publishers). Not real-time evidence.`;
    const clock = $("#clock");
    clock.innerHTML = Object.entries(health.demo.demo_clock_presets).map(([k, v]) => `<option value="${esc(v)}">${esc(k)}</option>`).join("");
    $("#clockWrap").hidden = false;
  }
}

async function ask(question, refresh = false) {
  lastQuestion = question;
  const btn = $("#askBtn");
  btn.disabled = true;
  const status = $("#status");
  status.hidden = false;
  status.className = "status";
  status.innerHTML = `<span class="spinner"></span>Understanding your question, resolving the fixture, gathering evidence and simulating lineups…`;
  const body = { question, timezone: tz(), refresh };
  if (health && health.mode === "demo") body.as_of = $("#clock").value;
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail ? JSON.stringify(data.detail) : res.statusText);
    render(data);
  } catch (err) {
    status.className = "status error";
    status.innerHTML = `<div class="msg">Request failed: ${esc(err.message)}</div>`;
    $("#result").hidden = true;
  } finally {
    btn.disabled = false;
  }
}

function render(r) {
  const status = $("#status");
  status.className = `status ${r.status}`;
  const notes = (r.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
  let extra = "";
  if ((r.status === "no_fixture" || r.status === "unsupported_competition") && r.next_fixture) {
    extra = `<div style="margin-top:10px"><button class="btn" id="predictNext">Forecast ${esc(r.next_fixture.home)} v ${esc(r.next_fixture.away)}</button></div>`;
  }
  const u = r.interpretation;
  const understood = u ? `<div class="muted" style="margin-top:6px">Understood by <b>${esc(u.backend)}</b>${u.cached ? " (cached)" : ""}:
    ${esc(u.intent || "")}${u.subject ? " · " + esc(u.subject) : ""}${u.opponent ? " vs " + esc(u.opponent) : ""}
    · ${esc(u.window || "")}${u.players && u.players.length ? " · " + esc(u.players_kind) + ": " + esc(u.players.join(", ")) : ""}</div>` : "";
  status.innerHTML = `<div class="msg">${esc(r.message)}</div>${understood}${notes ? `<ul>${notes}</ul>` : ""}${extra}`;
  const next = $("#predictNext");
  if (next) {
    // Name the opponent so the follow-up resolves to exactly this fixture.
    const opp = r.next_fixture.home === r.subject_team ? r.next_fixture.away : r.next_fixture.home;
    next.onclick = () => ask(`Will ${r.subject_team} win against ${opp}?`);
  }
  if (!r.forecast) {
    $("#result").hidden = !(r.fixture && r.data_quality && r.data_quality.length);
    if (!$("#result").hidden) {
      ["#matchCard", "#simCard", "#changeCard", "#lineupCard", "#newsCard", "#availCard", "#impactCard"].forEach((s) => ($(s).hidden = true));
      $("#whatIfCard").hidden = true;
      renderTransparency(r);
    }
    return;
  }
  ["#matchCard", "#simCard", "#changeCard", "#lineupCard", "#newsCard", "#availCard", "#impactCard"].forEach((s) => ($(s).hidden = false));
  $("#result").hidden = false;
  renderMatch(r);
  renderSimulations(r);
  renderChange(r);
  renderWhatIf(r);
  renderImpacts(r);
  renderLineups(r);
  renderNews(r);
  renderAvailability(r);
  renderTransparency(r);
  scheduleAutoRefresh(r);
}

function bar(p, thin = false) {
  const h = p.home_win, d = p.draw, a = p.away_win;
  const lab = (x) => (x > 0.07 ? pct(x) : "");
  return `<div class="bar${thin ? " thin" : ""}"><span class="h" style="width:${h * 100}%">${lab(h)}</span><span class="d" style="width:${d * 100}%">${lab(d)}</span><span class="a" style="width:${a * 100}%">${lab(a)}</span></div>`;
}

function renderMatch(r) {
  const f = r.fixture, fc = r.forecast, main = fc.adjusted || fc.baseline;
  const rows = [];
  if (fc.adjusted) rows.push(`<div class="probrow"><div class="lbl"><b>News-adjusted</b><br><span class="muted">provisional lineup layer</span></div>${bar(fc.adjusted.probabilities)}</div>`);
  rows.push(`<div class="probrow"><div class="lbl">Baseline ML<br><span class="muted">existing Poisson model</span></div>${bar(fc.baseline.probabilities, !!fc.adjusted)}</div>`);
  if (fc.logistic_comparison) rows.push(`<div class="probrow"><div class="lbl">Logistic comparison<br><span class="muted">not blended (val. weight 0%)</span></div>${bar(fc.logistic_comparison, true)}</div>`);
  const interval = main.interval_home_win ? `<div class="stat"><div class="k">90% scenario range (home win)</div><div class="val" style="font-size:15px">${pct(main.interval_home_win[0])} – ${pct(main.interval_home_win[1])}</div></div>` : "";
  $("#matchCard").innerHTML = `
    <div class="row"><h2>Match prediction</h2><span class="badge muted">${esc(f.competition.name)}${f.matchday ? " · MD " + f.matchday : ""}</span></div>
    <div class="teams"><span class="t">${esc(f.home)}</span><span class="v">vs</span><span class="t">${esc(f.away)}</span></div>
    ${hero(r)}
    <div class="meta">Kickoff ${esc(fmtTime(f.kickoff_utc))} (${esc(tz())}) · ${esc(f.status)} · source: ${esc(f.source)}</div>
    ${rows.join("")}
    <div class="legend"><span><i class="dot" style="background:var(--home)"></i>${esc(f.home)} win</span><span><i class="dot" style="background:var(--draw)"></i>Draw</span><span><i class="dot" style="background:var(--away)"></i>${esc(f.away)} win</span></div>
    <div class="stats">
      <div class="stat"><div class="k">Expected goals</div><div class="val">${main.expected_goals_home.toFixed(2)} – ${main.expected_goals_away.toFixed(2)}</div></div>
      <div class="stat"><div class="k">Baseline xG</div><div class="val" style="font-size:15px">${fc.baseline.expected_goals_home.toFixed(2)} – ${fc.baseline.expected_goals_away.toFixed(2)}</div></div>
      ${interval}
    </div>
    <div class="k muted" style="margin-top:14px">Most likely scorelines</div>
    <div class="scores">${main.top_scorelines.map((s) => `<span class="score">${s.home_goals}–${s.away_goals} · ${pct(s.probability, 1)}</span>`).join("")}</div>`;
}

function hero(r) {
  const sp = r.subject_probabilities;
  if (!sp) return "";
  return `<div class="hero"><div><div class="big">${pct(sp.win)}</div><div class="cap">${esc(sp.team)} win chance (${esc(sp.side)})</div></div>
    <div><div class="side">${pct(sp.draw)}</div><div class="cap">Draw</div></div>
    <div><div class="side">${pct(sp.lose)}</div><div class="cap">${esc(sp.opponent)} win</div></div></div>`;
}

const OUTCOME_COLORS = () => {
  const cs = getComputedStyle(document.documentElement);
  return { H: cs.getPropertyValue("--home").trim(), D: cs.getPropertyValue("--draw").trim(), A: cs.getPropertyValue("--away").trim() };
};

function renderSimulations(r) {
  const s = r.forecast.simulations, card = $("#simCard");
  if (!s) { card.hidden = true; return; }
  const f = r.fixture, sp = r.subject_probabilities;
  const subjectKey = sp && sp.side === "away" ? "away_win" : "home_win";
  const share = s.simulated_result_share;
  card.innerHTML = `<div class="row"><h2>Monte Carlo — all ${s.n.toLocaleString()} simulations</h2>
      <span><button class="btn ghost" id="toggleDraws">Show every draw</button> <button class="btn ghost" id="csvDraws">Download CSV</button></span></div>
    <p class="muted">${s.kind === "baseline" ? "No lineup layer available: every draw uses the baseline score grid; only the simulated scoreline varies." :
      "Each draw samples both starting XIs and the lineup-effect coefficients, then the existing Poisson/Dixon–Coles model gives that draw's win probabilities; one scoreline is then simulated from it."}
      The headline probabilities are the average over draws (exact), not the share of simulated scorelines below.</p>
    <div class="simgrid">
      <div class="simpanel"><h3>Every simulated result (one square per draw) — ${esc(f.home)} ${pct(share.H, 1)} · draw ${pct(share.D, 1)} · ${esc(f.away)} ${pct(share.A, 1)}</h3>
        <canvas class="draws" id="drawsCanvas"></canvas>
        <div class="legend"><span><i class="dot" style="background:var(--home)"></i>${esc(f.home)} win</span><span><i class="dot" style="background:var(--draw)"></i>Draw</span><span><i class="dot" style="background:var(--away)"></i>${esc(f.away)} win</span></div></div>
      <div class="simpanel"><h3>Simulated scorelines (${esc(f.home)} rows × ${esc(f.away)} columns)</h3><div id="heat"></div></div>
      <div class="simpanel"><h3>Distribution of ${esc(sp ? sp.team : f.home)} win probability across draws</h3><div id="hist"></div></div>
      <div class="simpanel"><h3>Goal rates per draw (λ ${esc(f.home)} vs λ ${esc(f.away)})</h3><canvas class="scatter" id="scatterCanvas"></canvas></div>
    </div>
    <div id="drawTable" class="tablewrap scroll" hidden style="margin-top:14px"></div>`;
  drawResults(s);
  drawHeat(s);
  drawHist(s[subjectKey]);
  drawScatter(s);
  $("#toggleDraws").onclick = () => {
    const t = $("#drawTable");
    t.hidden = !t.hidden;
    $("#toggleDraws").textContent = t.hidden ? "Show every draw" : "Hide draws";
    if (!t.hidden && !t.dataset.ready) {
      const rows = [];
      for (let i = 0; i < s.n; i++) rows.push(`<tr><td>${i + 1}</td><td>${pct(s.home_win[i], 1)}</td><td>${pct(s.draw[i], 1)}</td><td>${pct(s.away_win[i], 1)}</td><td>${s.lambda_home[i].toFixed(2)}</td><td>${s.lambda_away[i].toFixed(2)}</td><td>${s.score_home[i]}–${s.score_away[i]}</td></tr>`);
      t.innerHTML = `<table><tr><th>#</th><th>P(${esc(f.home)})</th><th>P(draw)</th><th>P(${esc(f.away)})</th><th>λ home</th><th>λ away</th><th>Simulated score</th></tr>${rows.join("")}</table>`;
      t.dataset.ready = "1";
    }
  };
  $("#csvDraws").onclick = () => {
    const lines = ["draw,p_home,p_draw,p_away,lambda_home,lambda_away,score_home,score_away"];
    for (let i = 0; i < s.n; i++) lines.push([i + 1, s.home_win[i], s.draw[i], s.away_win[i], s.lambda_home[i], s.lambda_away[i], s.score_home[i], s.score_away[i]].join(","));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = `matchsense_${f.home}_${f.away}_simulations.csv`.replace(/\s+/g, "_");
    a.click();
    URL.revokeObjectURL(a.href);
  };
}

function fitCanvas(canvas, height) {
  const ratio = window.devicePixelRatio || 1, width = canvas.clientWidth || 500;
  canvas.width = width * ratio; canvas.height = height * ratio; canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  return [ctx, width];
}

function drawResults(s) {
  const canvas = $("#drawsCanvas");
  const cols = 80, gap = 1;
  const width = canvas.clientWidth || 500, cell = (width - gap * cols) / cols;
  const rows = Math.ceil(s.n / cols);
  const [ctx] = fitCanvas(canvas, rows * (cell + gap));
  const c = OUTCOME_COLORS();
  for (let i = 0; i < s.n; i++) {
    const h = s.score_home[i], a = s.score_away[i];
    ctx.fillStyle = h > a ? c.H : h === a ? c.D : c.A;
    ctx.fillRect((i % cols) * (cell + gap), Math.floor(i / cols) * (cell + gap), cell, cell);
  }
}

function drawHeat(s) {
  const max = 5, counts = Array.from({ length: max + 1 }, () => Array(max + 1).fill(0));
  for (let i = 0; i < s.n; i++) counts[Math.min(s.score_home[i], max)][Math.min(s.score_away[i], max)]++;
  const top = Math.max(...counts.flat());
  let html = `<div class="heat" style="grid-template-columns:28px repeat(${max + 1},1fr)"><div class="hd"></div>`;
  for (let a = 0; a <= max; a++) html += `<div class="hd">${a}${a === max ? "+" : ""}</div>`;
  for (let h = 0; h <= max; h++) {
    html += `<div class="hd">${h}${h === max ? "+" : ""}</div>`;
    for (let a = 0; a <= max; a++) {
      const v = counts[h][a], alpha = top ? v / top : 0;
      html += `<div title="${h}–${a}: ${v} of ${s.n}" style="background:color-mix(in srgb, var(--accent) ${Math.round(alpha * 85)}%, var(--surface-2));color:${alpha > .55 ? "#fff" : "var(--text)"}">${v ? pct(v / s.n, 1) : ""}</div>`;
    }
  }
  $("#heat").innerHTML = html + "</div>";
}

function drawHist(values) {
  const lo = Math.min(...values), hi = Math.max(...values), bins = 30;
  const span = Math.max(hi - lo, 0.01), counts = Array(bins).fill(0);
  values.forEach((v) => counts[Math.min(bins - 1, Math.floor(((v - lo) / span) * bins))]++);
  const W = 420, H = 150, pad = 22, top = Math.max(...counts);
  const mean = values.reduce((x, y) => x + y, 0) / values.length;
  const bw = (W - pad * 2) / bins;
  const bars = counts.map((c, i) => `<rect x="${pad + i * bw}" y="${H - pad - (c / top) * (H - pad * 2)}" width="${bw - 1}" height="${(c / top) * (H - pad * 2)}" fill="var(--accent)" opacity=".8"><title>${pct(lo + (i * span) / bins, 1)}–${pct(lo + ((i + 1) * span) / bins, 1)}: ${c} draws</title></rect>`).join("");
  const mx = pad + ((mean - lo) / span) * (W - pad * 2);
  $("#hist").innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}">${bars}
    <line x1="${mx}" x2="${mx}" y1="${pad - 6}" y2="${H - pad}" stroke="var(--text)" stroke-dasharray="3 3"/>
    <text x="${mx + 4}" y="${pad - 8}">mean ${pct(mean, 1)}</text>
    <text x="${pad}" y="${H - 6}">${pct(lo, 1)}</text><text x="${W - pad}" y="${H - 6}" text-anchor="end">${pct(hi, 1)}</text></svg>`;
}

function drawScatter(s) {
  const canvas = $("#scatterCanvas");
  const [ctx, width] = fitCanvas(canvas, 170);
  const H = 170, pad = 24;
  const xs = s.lambda_home, ys = s.lambda_away;
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (v) => pad + ((v - x0) / Math.max(x1 - x0, 0.01)) * (width - pad * 2);
  const sy = (v) => H - pad - ((v - y0) / Math.max(y1 - y0, 0.01)) * (H - pad * 2);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  ctx.globalAlpha = 0.35;
  for (let i = 0; i < s.n; i++) { ctx.beginPath(); ctx.arc(sx(xs[i]), sy(ys[i]), 2, 0, Math.PI * 2); ctx.fill(); }
  ctx.globalAlpha = 1;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  ctx.font = "10px sans-serif";
  ctx.fillText(`λ home ${x0.toFixed(2)}–${x1.toFixed(2)}`, pad, H - 6);
  ctx.fillText(`λ away ${y0.toFixed(2)}–${y1.toFixed(2)}`, pad, 12);
}

function renderImpacts(r) {
  const rows = r.forecast.player_impacts || [], card = $("#impactCard");
  if (!rows.length) { card.hidden = true; return; }
  const body = rows.map((x) => {
    const pts = x.delta_win * 100, cls = pts <= -1 ? "bad" : pts >= 1 ? "" : "muted";
    return `<tr><td><b>${esc(x.short_name || x.player)}</b> <span class="muted">${esc(x.position)}</span></td><td>${esc(x.team)}</td>
      <td>${pct(x.p_start)}</td><td>${pct(x.goal_share)}</td>
      <td><span class="badge ${cls}">${pts >= 0 ? "+" : ""}${pts.toFixed(1)} pts</span></td>
      <td class="muted">${x.delta_xg_for >= 0 ? "+" : ""}${x.delta_xg_for.toFixed(3)} / ${x.delta_xg_against >= 0 ? "+" : ""}${x.delta_xg_against.toFixed(3)}</td>
      <td><button class="btn ghost" data-player="${esc(x.short_name || x.player)}">What if out?</button></td></tr>`;
  }).join("");
  card.innerHTML = `<h2>Player impact — how much each likely starter is worth in this model</h2>
    <p class="muted">Change in that team's win chance if the player is ruled out and the model re-picks the XI, all else fixed.
      Attacking value comes from each player's share of recent goals. Defenders count through how established they are and whether the usual
      goalkeeper/back line is broken (raising the opponent's expected goals); their <b>individual</b> defensive quality is not modelled, because one
      season of data could not identify it. Rotation players can show ~0 when their replacement is the more regular starter.</p>
    <div class="tablewrap scroll"><table><tr><th>Player</th><th>Team</th><th>Start chance</th><th>Share of team goals</th><th>Team win chance if out</th><th>xG for / against</th><th></th></tr>${body}</table></div>`;
  card.querySelectorAll("button[data-player]").forEach((b) => b.onclick = () => {
    const q = `What are the odds if ${b.dataset.player} doesn't play?`;
    $("#question").value = q; ask(q); window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function renderChange(r) {
  const fc = r.forecast;
  if (!fc.adjusted) {
    $("#changeCard").innerHTML = `<h2>What changed?</h2><p class="muted">No adjustment applied — the forecast is the baseline model.</p>${fc.explanation.map((e) => `<p class="muted">${esc(e)}</p>`).join("")}`;
    return;
  }
  const b = fc.baseline.probabilities, a = fc.adjusted.probabilities;
  const deltas = [["Home win", a.home_win - b.home_win, "var(--home)"], ["Draw", a.draw - b.draw, "var(--draw)"], ["Away win", a.away_win - b.away_win, "var(--away)"]];
  const max = Math.max(0.05, ...deltas.map((d) => Math.abs(d[1])));
  const drow = ([lab, d, col]) => {
    const w = (Math.abs(d) / max) * 50;
    const left = d >= 0 ? 50 : 50 - w;
    return `<div class="delta"><span>${lab}</span><div class="track"><div class="fill" style="left:${left}%;width:${w}%;background:${col}"></div></div><span class="num">${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} pts</span></div>`;
  };
  const contribs = fc.contributions.slice(0, 8).map((c) => `<div class="contrib"><span>${esc(c.factor)} <span class="muted">(${esc(c.team)})</span></span><span class="x">xG for ${c.delta_xg_for_team >= 0 ? "+" : ""}${c.delta_xg_for_team.toFixed(2)} · against ${c.delta_xg_against_team >= 0 ? "+" : ""}${c.delta_xg_against_team.toFixed(2)}</span></div>`).join("");
  $("#changeCard").innerHTML = `<h2>What changed? Baseline → adjusted</h2>${deltas.map(drow).join("")}
    <div style="margin-top:12px">${contribs || '<p class="muted">No material lineup factors.</p>'}</div>
    ${fc.explanation.map((e) => `<p class="muted" style="margin:8px 0 0">${esc(e)}</p>`).join("")}
    <div class="note">${esc(fc.adjustment_status)}</div>`;
}

function renderWhatIf(r) {
  const w = r.what_if, card = $("#whatIfCard");
  $("#changeCard").classList.toggle("span2", !w);
  if (!w) { card.hidden = true; return; }
  card.hidden = false;
  if (w.error) { card.innerHTML = `<h2>What if…</h2><p>${esc(w.error)}</p>`; return; }
  const s = w.scenarios;
  card.innerHTML = `<h2>What if: ${esc(w.players.join(", "))}</h2>
    <div class="probrow"><div class="lbl">Available</div>${bar(s.available.probabilities)}</div>
    <div class="probrow"><div class="lbl">Unavailable</div>${bar(s.unavailable.probabilities)}</div>
    <p>${esc(w.summary)}</p>${w.replacements && w.replacements.length ? `<p class="muted">Most likely to come in: ${w.replacements.map((x) => `${esc(x.player)} (+${Math.round(x.start_gain * 100)} pts start chance)`).join(", ")}</p>` : ""}${w.note ? `<p class="muted">${esc(w.note)}</p>` : ""}${w.unresolved && w.unresolved.length ? `<p class="muted">Not matched: ${esc(w.unresolved.join(", "))}</p>` : ""}`;
}

function pitchSvg(lineup, flip) {
  const W = 300, H = 400;
  const lines = { GK: 365, DEF: 285, MID: 190, FWD: 95 };
  const groups = { GK: [], DEF: [], MID: [], FWD: [] };
  lineup.starters.forEach((s) => groups[s.position].push(s));
  if (!groups.FWD.length && groups.MID.length > 4) { lines.MID = 215; lines.FWD = 0; }
  let players = "";
  for (const [pos, list] of Object.entries(groups)) {
    list.forEach((s, i) => {
      const x = (W / (list.length + 1)) * (i + 1);
      const y = flip ? H - lines[pos] : lines[pos];
      const confirmed = lineup.kind === "confirmed";
      const strong = confirmed || s.p_start >= 0.8;
      const last = s.short_name || s.name.split(" ").slice(-1)[0];
      const surname = list.length >= 5 && last.length > 9 ? last.slice(0, 8) + "." : last;
      players += `<g class="pl"><circle cx="${x}" cy="${y}" r="15" fill="${strong ? "rgba(255,255,255,.92)" : "rgba(255,255,255,.35)"}" stroke="#fff" stroke-width="2" ${s.uncertain ? 'stroke-dasharray="4 3"' : ""}/>
        <text x="${x}" y="${y + 30}"${list.length >= 5 ? ' style="font-size:9.5px"' : ""}>${esc(surname)}</text>${confirmed ? "" : `<text class="pct" x="${x}" y="${y + 4}" style="fill:#14301f;stroke:none">${Math.round(s.p_start * 100)}</text>`}</g>`;
    });
  }
  return `<svg class="pitch" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(lineup.team)} lineup">
    <rect width="${W}" height="${H}" fill="var(--pitch)"/>
    <g fill="none" stroke="var(--pitch-line)" stroke-width="1.5"><rect x="8" y="8" width="${W - 16}" height="${H - 16}"/>
    <line x1="8" y1="${H / 2}" x2="${W - 8}" y2="${H / 2}"/><circle cx="${W / 2}" cy="${H / 2}" r="38"/>
    <rect x="${W / 2 - 70}" y="8" width="140" height="58"/><rect x="${W / 2 - 70}" y="${H - 66}" width="140" height="58"/></g>${players}</svg>`;
}

function renderLineups(r) {
  const teams = [r.fixture.home, r.fixture.away];
  const blocks = teams.map((t, idx) => {
    const l = r.lineups[t];
    if (!l) return `<div><div class="pitchhead"><h3>${esc(t)}</h3><span class="badge bad">unavailable</span></div><p class="muted">No squad data — baseline only.</p></div>`;
    const badge = l.kind === "confirmed" ? '<span class="badge">Confirmed XI</span>' : '<span class="badge warn">Predicted — not confirmed</span>';
    const bench = l.bench.slice(0, 7).map((s) => `${esc(s.short_name || s.name)} ${l.kind === "confirmed" ? "" : pct(s.p_start)}`).join(", ");
    const excl = l.excluded.map((s) => `<b>${esc(s.name)}</b> (${esc(s.status)})`).join(", ");
    return `<div><div class="pitchhead"><h3>${esc(t)} <span class="muted">${esc(l.formation || "")}</span></h3>${badge}</div>
      ${pitchSvg(l, idx === 1)}
      <div class="benchlist">${l.kind === "confirmed" ? "" : "Next most likely: "}${bench || "—"}</div>
      ${excl ? `<div class="benchlist">Excluded: ${excl}</div>` : ""}
      <div class="benchlist">${esc(l.note)}</div></div>`;
  });
  $("#lineupCard").innerHTML = `<div class="row"><h2>Starting XIs</h2><span class="muted">Numbers = % of simulated lineups in which the player starts · dashed = uncertain</span></div><div class="pitches">${blocks.join("")}</div>`;
}

function certaintyBadge(c) {
  const map = { confirmed_fact: ["Confirmed", ""], reported_expectation: ["Reported", "muted"], speculation: ["Speculation", "warn"], ai_inference: ["AI inference", "warn"] };
  const [t, cls] = map[c] || [c, "muted"];
  return `<span class="badge ${cls}">${t}</span>`;
}

function renderNews(r) {
  const items = r.news.filter((e) => e.category !== "other");
  const now = r.generated_at;
  const html = items.map((e) => {
    const url = safeUrl(e.url);
    const src = url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(e.source)}</a>` : esc(e.source);
    return `<div class="ev"><div class="top"><span class="who">${esc(e.entity)}</span><span class="muted">${esc(e.team || "")}</span>
      <span class="badge muted">${esc(e.event_type.replace(/_/g, " "))}</span>${certaintyBadge(e.certainty)}
      <span class="badge ${e.availability === "unavailable" ? "bad" : e.availability === "doubtful" ? "warn" : "muted"}">${esc(e.availability)}</span>
      ${e.is_simulated ? '<span class="badge demo">simulated</span>' : ""}</div>
      <div class="quote">“${esc(e.evidence)}”</div>
      <div class="src">${src} · published ${esc(fmtTime(e.published_at))} (${esc(ago(e.published_at, now))}) · retrieved ${esc(fmtTime(e.retrieved_at))} · credibility ${e.credibility.toFixed(2)} · ${esc(e.extractor)}${e.match_relevance ? " · relevance " + esc(e.match_relevance) : ""}</div></div>`;
  }).join("");
  $("#newsCard").innerHTML = `<h2>News intelligence (last 72h + current flags)</h2><div class="scroll">${html || '<p class="muted">No relevant structured evidence found.</p>'}</div>`;
}

function renderAvailability(r) {
  const rows = r.availability.map((a) => `<tr><td><b>${esc(a.name)}</b><br><span class="muted">${esc(a.team)}</span></td>
    <td>${pct(a.p_available)}</td><td><span class="badge ${a.status === "unavailable" ? "bad" : a.status === "doubtful" ? "warn" : ""}">${esc(a.status)}</span></td>
    <td>${certaintyBadge(a.basis)}${a.conflict ? ' <span class="badge bad">conflict</span>' : ""}<div class="muted">${esc(a.note)}</div></td></tr>`).join("");
  $("#availCard").innerHTML = `<h2>Corroborated availability</h2>
    <p class="muted">Credibility- and freshness-weighted beliefs; syndicated copies count once; newer confirmed statements override reports.</p>
    <div class="tablewrap scroll"><table><tr><th>Player</th><th>P(avail.)</th><th>Status</th><th>Basis</th></tr>${rows || '<tr><td colspan="4" class="muted">No availability evidence.</td></tr>'}</table></div>`;
}

function renderTransparency(r) {
  const card = $("#transparencyCard");
  card.hidden = false;
  const rows = (r.data_quality || []).map((d) => `<tr><td>${esc(d.name)}</td><td><span class="st st-${esc(d.status)}">${esc(d.status)}</span></td><td>${esc(d.detail)}</td><td class="muted">${d.as_of ? esc(fmtTime(d.as_of)) : ""}</td></tr>`).join("");
  card.innerHTML = `<div class="row"><h2>Model transparency &amp; data freshness</h2><button class="btn ghost" id="refreshBtn">↻ Refresh pipeline</button></div>
    <p class="muted">Model: ${esc(r.model_version || "—")}<br>Prediction generated ${esc(fmtTime(r.generated_at))} · mode ${esc(r.mode)}${r.prediction_id ? " · id " + esc(r.prediction_id) : ""}</p>
    <div class="tablewrap"><table><tr><th>Input</th><th>Status</th><th>Detail</th><th>As of</th></tr>${rows}</table></div>`;
  $("#refreshBtn").onclick = () => lastQuestion && ask(lastQuestion, true);
}

function scheduleAutoRefresh(r) {
  clearTimeout(autoTimer);
  if (!r.fixture || r.mode === "demo") return;
  const minutes = (new Date(r.fixture.kickoff_utc) - Date.now()) / 6e4;
  const confirmed = Object.values(r.lineups || {}).every((l) => l.kind === "confirmed");
  // Official XIs land ~60 min before kickoff: poll so they replace predictions automatically.
  if (minutes > 0 && minutes < 90 && !confirmed) autoTimer = setTimeout(() => ask(lastQuestion, true), 5 * 60 * 1000);
}

async function loadHistory() {
  const h = await fetch("/api/history").then((r) => r.json());
  const s = h.scoring;
  const fmt = (x) => (x == null ? "—" : x.toFixed(4));
  $("#scoring").innerHTML = `<p>Scored pre-kickoff forecasts: baseline n=${s.baseline.n}, log loss ${fmt(s.baseline.log_loss)}, Brier ${fmt(s.baseline.brier)} · adjusted n=${s.adjusted.n}, log loss ${fmt(s.adjusted.log_loss)}, Brier ${fmt(s.adjusted.brier)}</p>`;
  $("#historyTable").innerHTML = `<tr><th>Issued</th><th>Match</th><th>Kickoff</th><th>Lineups</th><th>Baseline H/D/A</th><th>Adjusted H/D/A</th><th>Result</th><th>Mode</th></tr>` +
    h.predictions.map((p) => `<tr><td>${esc(fmtTime(p.as_of))}</td><td>${esc(p.home)} v ${esc(p.away)}</td><td>${esc(fmtTime(p.kickoff_utc))}</td><td>${esc(p.lineup_kind)}</td>
      <td>${pct(p.baseline_home)} / ${pct(p.baseline_draw)} / ${pct(p.baseline_away)}</td><td>${pct(p.adjusted_home)} / ${pct(p.adjusted_draw)} / ${pct(p.adjusted_away)}</td>
      <td>${p.home_goals == null ? "—" : `${p.home_goals}–${p.away_goals}`}</td><td>${esc(p.mode)}</td></tr>`).join("");
}

document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  $("#view-ask").hidden = btn.dataset.view !== "ask";
  $("#view-history").hidden = btn.dataset.view !== "history";
  if (btn.dataset.view === "history") loadHistory();
}));
$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#question").value.trim(); if (q) ask(q); });
$("#examples").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") { $("#question").value = e.target.textContent; ask(e.target.textContent); } });
$("#tz").addEventListener("change", () => { try { localStorage.setItem("ms-tz", tz()); } catch (_) {} if (lastQuestion) ask(lastQuestion); });
$("#clock").addEventListener("change", () => { if (lastQuestion) ask(lastQuestion); });
loadHealth();
