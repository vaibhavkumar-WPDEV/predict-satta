"use strict";

const state = { dash: null, market: null, tab: "today", mode: "server", month: null };

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "–" : (x * 100).toFixed(d) + "%");
const jd = (v) => (v == null ? "XX" : String(v).padStart(2, "0"));
const pv = (p) => (p == null ? "" : p < 0.001 ? "p<0.001" : "p=" + p.toFixed(3));

async function load() {
  let dash = null;
  try {
    const r = await fetch("api/dashboard", { cache: "no-store" });
    if (r.ok) dash = await r.json();
  } catch (_) { /* static mode */ }
  if (!dash) {
    state.mode = "static";
    for (const url of ["../data/dashboard.json", "data/dashboard.json"]) {
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (r.ok) { dash = await r.json(); break; }
      } catch (_) { /* try next */ }
    }
  }
  if (!dash) {
    $("#meta").textContent = "Dashboard data nahi mila. Server chalao: python -m satta serve";
    return;
  }
  state.dash = dash;
  if (!state.market) state.market = dash.primary;
  render();
}

function render() {
  const d = state.dash;
  const sync = d.sync || {};
  let syncTxt = "";
  if (sync.error) syncTxt = " · fetch error";
  else if (sync.sources) syncTxt = " · sources: " + sync.sources.map((s) => `${s.source} ${s.values}`).join(", ");
  $("#meta").textContent = `Updated ${fmtTime(d.generated_at)}${syncTxt}${state.mode === "static" ? " · static mode" : ""}`;
  $("#refresh").style.display = state.mode === "static" ? "none" : "";
  renderMarketBar();
  renderToday();
  renderProof();
  renderTest();
  renderLearning();
  renderFormulas();
  renderTheorems();
  renderChart();
}

function fmtTime(iso) {
  if (!iso) return "–";
  const t = new Date(iso);
  return t.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) + " IST";
}

function renderMarketBar() {
  const bar = $("#marketBar");
  const show = !["today", "chart", "help"].includes(state.tab);
  bar.style.display = show ? "flex" : "none";
  bar.innerHTML = Object.values(state.dash.markets)
    .map((m) => `<button data-m="${m.key}" class="${m.key === state.market ? "active" : ""}">${esc(m.name)} (${esc(m.short)})</button>`)
    .join("");
  bar.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.market = b.dataset.m; render(); }));
}

const cur = () => state.dash.markets[state.market];

// ------------------------------------------------------------------ today
function predictionCard(m, hero) {
  if (!m.ready) {
    return `<div class="card"><h2>${esc(m.name)}</h2><p class="muted">${esc(m.reason)}</p></div>`;
  }
  const p = m.next;
  if (!p) return `<div class="card"><h2>${esc(m.name)}</h2><p class="muted">Prediction abhi lock nahi hui.</p></div>`;
  const maxP = p.top10[0][1];
  const jodis = p.top10.map(([v, pr], i) => `
      <div class="jodi ${i === 0 ? "first" : ""}"><b>${jd(v)}</b><small>${pct(pr, 2)}</small>
      <div class="bar"><i style="width:${(pr / maxP) * 100}%"></i></div></div>`).join("");
  const bt = m.backtest.all;
  const lv = m.live.summary;
  const formulas = (p.formulas || []).map((f) => `<div class="formula">${esc(f)}</div>`).join("");
  return `
  <div class="card ${hero ? "hero" : ""}">
    <div class="row" style="justify-content:space-between">
      <h2>${esc(m.name)} <span class="muted">(${esc(m.short)})</span></h2>
      <span class="pill ${p.late ? "bad" : "good"}">${p.late ? "LATE (result ke baad bani)" : "LOCKED before result"}</span>
    </div>
    <div class="row"><span class="big">Result date</span> <b>${esc(p.date)}</b>
      <span class="big">· time ~${esc(m.result_time)} IST</span></div>
    ${hero ? `<div class="countdown" data-until="${esc(p.result_time)}"></div>` : ""}
    ${(() => {
      const r = lastEvaluated(m);
      return r ? `<div class="muted small">Pichla result (${esc(r.date)}): <b>${jd(r.actual)}</b> ${resultPill(r)}</div>` : "";
    })()}
    ${(m.closed_days || []).length ? `<div class="muted" style="font-size:12px">Data se seekha: ${m.closed_days.map((k) => ({ month_end: "mahine ke aakhri din", month_start: "mahine ki 1 tareekh" }[k] || k)).join(", ")} result nahi aata — us din ki prediction nahi banti.</div>` : ""}
    <div class="jodis">${jodis}</div>
    <div class="digits">
      <span>Andar: ${p.andar.map(([d, pr]) => `<b>${d}</b><small class="muted">${pct(pr, 0)}</small>`).join(" ")}</span>
      <span>Bahar: ${p.bahar.map(([d, pr]) => `<b>${d}</b><small class="muted">${pct(pr, 0)}</small>`).join(" ")}</span>
    </div>
    ${formulas ? `<h3>Tool ke formule (is din ke liye)</h3>${formulas}` : ""}
    <h3>Proof</h3>
    <div class="muted" style="font-size:12px">Locked: ${fmtTime(p.created_at)} · ${p.trained_on} results par trained</div>
    <div class="mono muted">SHA-256 ${esc(p.hash)}</div>
    <h3>Kitna close? (model ka dawa vs asli test)</h3>
    <div class="row">
      <span class="pill warn">Model ka dawa: top-10 me aane ka ${pct(p.top10.reduce((s, x) => s + x[1], 0))}, #1 number ka ${pct(p.top10[0][1], 2)}</span>
      <span class="pill info">Asli test (bina result dekhe, ${bt.n || 0} din): top-10 ${pct(bt.hit10?.rate)}, exact ${pct(bt.hit1?.rate)} ${pv(bt.hit10?.p_value)}</span>
      <span class="pill info">Live top-10: ${lv.n ? pct(lv.hit10.rate) + ` (${lv.hit10.hits}/${lv.n})` : "abhi data nahi"}</span>
    </div>
  </div>`;
}

function renderToday() {
  const d = state.dash;
  const ms = Object.values(d.markets);
  const primary = d.markets[d.primary];
  const others = ms.filter((m) => m.key !== d.primary);
  let banner = "";
  if (primary && primary.ready) {
    const h = primary.backtest.all.hit10;
    if (h) {
      const better = h.p_value < 0.05;
      banner = `<div class="banner ${better ? "" : "bad"}">
        <b>${esc(primary.name)} — honest check:</b> pichle ${primary.backtest.all.n} din ke walk-forward test me
        Top-10 me asli number <b>${pct(h.rate)}</b> baar aaya; random guess se <b>10%</b> aata. ${pv(h.p_value)}.
        ${better ? "Tool random se behtar dikh raha hai." : "Abhi tak tool random chance se behtar sabit nahi hua — numbers ko guarantee mat samjho."}
        ${primary.selfbreak ? `<br>"Khud ko todo" test: asli data par top-10 ${pct(primary.selfbreak.real_hit10)}, shuffled (pattern-less) data par ${pct(primary.selfbreak.shuffled_hit10)} — ${esc(primary.selfbreak.verdict)}.` : ""}
      </div>`;
    }
  } else if (!ms.some((m) => m.ready)) {
    const srcs = (d.sync && d.sync.sources) || [];
    const allFailed = srcs.length && srcs.every((s) => s.values === 0);
    banner = `<div class="banner bad"><b>Data nahi hai.</b> ${state.mode === "server" ? '"Abhi Update Karo" dabao ya thoda ruko, tool khud websites se data la raha hai.' : "Server/GitHub Action abhi tak nahi chala."}
      ${d.sync && d.sync.error ? "<br>Error: " + esc(d.sync.error) : ""}
      ${allFailed ? "<br>Koi bhi result website nahi khuli (internet/firewall check karo). Sources: " + srcs.map((s) => esc(s.source)).join(", ") : ""}</div>`;
  }
  $("#tab-today").innerHTML = statusStrip(d) + banner + (primary ? predictionCard(primary, true) : "") +
    (primary ? coverageCard(primary) : "") +
    `<div class="grid">${others.map((m) => predictionCard(m, false)).join("")}</div>`;
  const rate = $("#payRate");
  if (rate) rate.addEventListener("change", (e) => { state.rate = Number(e.target.value) || 90; renderToday(); });
  tickCountdown();
}

const todayIST = () => new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });

function lastEvaluated(m) {
  return ((m.live && m.live.rows) || []).find((r) => r.actual != null);
}

function resultPill(r) {
  if (!r || r.actual == null) return "";
  return r.status === "hit"
    ? `<span class="pill good">HIT · rank ${r.rank}</span>`
    : `<span class="pill bad">MISS · rank ${r.rank}</span>`;
}

// "Aaj ke results": when each market's result comes and what came.
function statusStrip(d) {
  const today = todayIST();
  const ms = Object.values(d.markets).filter((m) => m.ready)
    .sort((a, b) => a.result_time.localeCompare(b.result_time));
  if (!ms.length) return "";
  const cells = ms.map((m) => {
    const row = ((m.live && m.live.rows) || []).find((r) => r.date === today);
    const res = d.results.find((r) => r.date === today);
    const actual = row && row.actual != null ? row.actual : res ? res[m.key] : null;
    let body;
    if (actual != null) {
      body = `<div class="big">${jd(actual)}</div><div>Result aa gaya ${row ? resultPill(row) : ""}</div>
        ${row ? `<div class="muted small">Locked top-10: ${row.top10.map(jd).join(" ")}</div>` : ""}`;
    } else if (m.next && m.next.date === today) {
      body = `<div class="big">⏳</div><div>~${esc(m.result_time)} IST</div>
        <div class="small" data-until="${esc(m.next.result_time)}"></div>`;
    } else {
      body = `<div class="big">–</div><div class="muted">Aaj result nahi (chhutti) · agla ${esc(m.next ? m.next.date : "")}</div>`;
    }
    return `<div class="status-cell"><div class="row" style="justify-content:space-between">
      <b>${esc(m.name)}</b><span class="muted small">${esc(m.result_time)} IST</span></div>${body}</div>`;
  }).join("");
  return `<div class="card"><h2>Aaj ke results (${esc(today)})</h2>
    <p class="muted small">Result aate hi tool khud laata hai (har 30 minute check) aur locked prediction se milata hai. Poori list: Proof (Live) tab.</p>
    <div class="status-grid">${cells}</div></div>`;
}

// Ranking that starts with the locked list (exact order) and continues by probability.
function ranked(locked, probs, n) {
  const seen = new Set(locked.map(([v]) => v));
  const rest = probs.map((p, v) => [v, p]).filter(([v]) => !seen.has(v))
    .sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return locked.concat(rest).slice(0, n);
}

// "Percentage kaise badhe": the only honest lever is covering more numbers.
function coverageCard(m) {
  const c = m.coverage;
  const p = m.next;
  if (!m.ready || !c || !p || !p.dist) return "";
  const rate = state.rate || 90;
  const rows = c.jodi.map((r) => {
    const top = ranked(p.top10, p.dist, r.n);
    const claim = top.reduce((s, x) => s + x[1], 0);
    const back = r.tested * rate * 100 / r.n;
    const list = top.map(([v]) => jd(v)).join(" ");
    return `<tr>
      <td class="num"><b>${r.n}</b></td>
      <td class="mono">${r.n <= 10 ? list : `<details><summary>${r.n} numbers dekho</summary>${list}</details>`}</td>
      <td class="num">${pct(claim)}</td>
      <td class="num"><b>${pct(r.tested)}</b><br><small class="muted">200 din: ${pct(r.tested_200)}</small></td>
      <td class="num">${pct(r.random, 0)}</td>
      <td class="num">₹${back.toFixed(0)}</td></tr>`;
  }).join("");
  const at = [], ab = [];
  p.dist.forEach((x, v) => { at[Math.floor(v / 10)] = (at[Math.floor(v / 10)] || 0) + x; ab[v % 10] = (ab[v % 10] || 0) + x; });
  const digitRows = (arr, tested, name) => tested.map((r) => {
    const top = ranked(name === "Andar" ? p.andar : p.bahar, arr, r.k);
    return `<tr><td>${name} top-${r.k}</td><td class="mono">${top.map(([d]) => d).join(", ")}</td>
      <td class="num"><b>${pct(r.tested)}</b></td><td class="num">${pct(r.random, 0)}</td></tr>`;
  }).join("");
  return `
  <div class="card">
    <h2>Percentage kaise badhe? Jitne zyada numbers, utna zyada chance</h2>
    <p class="muted">${esc(m.name)} ke agle result (${esc(p.date)}) ke liye tool ki list. "Asli test" = pichle ${c.days} din me bina result dekhe tool ki top-N list me asli number kitni baar aaya.</p>
    <div class="table-wrap"><table>
      <thead><tr><th class="num">Kitne numbers</th><th>Tool ke numbers</th><th class="num">Model ka dawa</th>
      <th class="num">Asli test</th><th class="num">Random</th>
      <th class="num">₹100 lagane par ausatan wapas<br><small class="muted">rate
        <input id="payRate" type="number" min="1" max="200" value="${rate}" style="width:52px"> guna</small></th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <h3>Andar / Bahar</h3>
    <div class="table-wrap"><table>
      <thead><tr><th></th><th>Digits</th><th class="num">Asli test</th><th class="num">Random</th></tr></thead>
      <tbody>${digitRows(at, c.andar, "Andar")}${digitRows(ab, c.bahar, "Bahar")}</tbody></table></div>
    <p class="muted" style="font-size:12px">Hisaab: N numbers par ₹100 barabar baantne se har number par ₹100/N lagta hai; asli number list me aaya to ₹100/N × rate milta hai. Ausatan wapas = asli test % × rate × 100 / N. Random picking me yeh hamesha ₹${rate} hota hai, chahe N kitna bhi ho.</p>
  </div>`;
}

function tickCountdown() {
  document.querySelectorAll("[data-until]").forEach((el) => {
    const ms = new Date(el.dataset.until) - new Date();
    if (ms <= 0) { el.textContent = "Result time ho gaya — result aate hi tool khud check karega"; return; }
    const h = Math.floor(ms / 3.6e6), m = Math.floor((ms % 3.6e6) / 6e4), s = Math.floor((ms % 6e4) / 1e3);
    el.textContent = `Result me ${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
  });
}
setInterval(tickCountdown, 1000);

// ------------------------------------------------------------------ proof
function summaryStats(s, title) {
  if (!s || !s.n) return `<div class="stat"><div class="s">${esc(title)}</div><div class="v">–</div><div class="s">abhi data nahi</div></div>`;
  const item = (k, name) => `<div class="stat"><div class="s">${name}</div>
      <div class="v">${pct(s[k].rate)}</div>
      <div class="s">${s[k].hits}/${s.n} · random ${pct(s[k].chance, 0)} · ${pv(s[k].p_value)}</div></div>`;
  return `<h3>${esc(title)} (${s.n} din)</h3><div class="grid stats">
    ${item("hit1", "Exact number")}${item("hit10", "Top-10 me")}${item("andar_hit", "Andar top-3")}${item("bahar_hit", "Bahar top-3")}
    <div class="stat"><div class="s">Average rank of actual</div><div class="v">${s.mean_rank.value.toFixed(1)}</div>
    <div class="s">random 50.5 · ${pv(s.mean_rank.p_value)}</div></div></div>`;
}

function whyList(lines) {
  if (!lines || !lines.length) return "";
  return `<details><summary>kyu? / kya seekha</summary><ul class="why">${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></details>`;
}

function renderProof() {
  const m = cur();
  if (!m.ready) { $("#tab-proof").innerHTML = `<div class="card">${esc(m.reason)}</div>`; return; }
  const rows = m.live.rows.map((r) => {
    const status = { hit: '<span class="pill good">HIT</span>', miss: '<span class="pill bad">MISS</span>',
      pending: '<span class="pill warn">result ka intezaar</span>', "no-result": '<span class="pill">result nahi aaya</span>' }[r.status];
    return `<tr class="${r.status}">
      <td>${esc(r.date)}</td>
      <td>${fmtTime(r.created_at)} ${r.late ? '<span class="pill bad">late</span>' : ""}</td>
      <td class="mono">${r.top10.map(jd).join(" ")}</td>
      <td class="num"><b>${jd(r.actual)}</b></td>
      <td class="num">${r.rank ?? "–"}</td>
      <td>${status} ${whyList(r.why)}</td>
      <td class="mono" title="${esc(r.hash)}">${r.verified ? "✔" : "✘ TAMPERED"} ${esc(r.hash.slice(0, 10))}…</td>
    </tr>`;
  }).join("");
  $("#tab-proof").innerHTML = `
    <div class="card"><h2>${esc(m.name)}: live predictions (result se pehle lock)</h2>
      <p class="muted">Sirf wahi predictions gini jaati hain jo result time se pehle lock hui aur jinka hash match karta hai.</p>
      ${summaryStats(m.live.summary, "Live accuracy")}
    </div>
    <div class="card table-wrap"><table>
      <thead><tr><th>Date</th><th>Lock time</th><th>Top-10</th><th class="num">Actual</th><th class="num">Rank</th><th>Status</th><th>Hash</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="7" class="muted">Abhi koi live prediction nahi.</td></tr>'}</tbody>
    </table></div>`;
}

// ------------------------------------------------------------------- test
function renderTest() {
  const m = cur();
  if (!m.ready) { $("#tab-test").innerHTML = `<div class="card">${esc(m.reason)}</div>`; return; }
  const bt = m.backtest;
  const rows = bt.rows.map((r) => `<tr class="${r.hit10 ? "hit" : "miss"}">
      <td>${esc(r.date)}</td><td class="num"><b>${jd(r.actual)}</b></td>
      <td class="mono">${r.top10.map((v) => (v === r.actual ? `<span class="hl">${jd(v)}</span>` : jd(v))).join(" ")}</td>
      <td class="num">${r.rank}</td>
      <td>${r.hit10 ? '<span class="pill good">HIT</span>' : '<span class="pill bad">MISS</span>'}
        ${r.andar_hit ? '<span class="pill">A✔</span>' : ""}${r.bahar_hit ? '<span class="pill">B✔</span>' : ""}
        ${whyList(r.why)}</td></tr>`).join("");
  $("#tab-test").innerHTML = `
    <div class="card"><h2>${esc(m.name)}: walk-forward test</h2>
      <p class="muted">Har din ki prediction sirf us din se pehle ke data se bani (future leak nahi) — bilkul live jaisa.</p>
      ${summaryStats(bt.last7, "Pichle 7 din")}
      ${summaryStats(bt.last30, "Pichle 30 din")}
      ${summaryStats(bt.last100, "Pichle 100 din")}
      ${summaryStats(bt.last200, "Pichle 200 din")}
      ${summaryStats(bt.all, "Poora itihaas")}
    </div>
    <div class="card table-wrap"><table>
      <thead><tr><th>Date</th><th class="num">Actual</th><th>Tool ki Top-10</th><th class="num">Rank</th><th>Result</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

// --------------------------------------------------------------- learning
function renderLearning() {
  const m = cur();
  if (!m.ready) { $("#tab-learning").innerHTML = `<div class="card">${esc(m.reason)}</div>`; return; }
  const rows = m.experts.map((e) => `<tr>
      <td><b>${esc(e.label)}</b><div class="muted formula" style="font-size:12px">${esc(e.theory)}</div></td>
      <td class="num">${pct(e.weight)}</td><td class="num">${e.mean_rank ?? "–"}</td>
      <td class="num">${pct(e.hit10_rate)}</td><td class="num">${e.avg_logloss ?? "–"}</td></tr>`).join("");
  const pg = m.progress;
  const sb = m.selfbreak;
  const cmp = (s) => (s && s.n ? `${pct(s.hit10.rate)} top-10 · avg rank ${s.mean_rank.value.toFixed(1)} · log-loss ${s.logloss.value.toFixed(3)}` : "–");
  const progressCard = pg ? `
    <div class="card"><h2>Seekhne ka asar: kitna sudhra?</h2>
      <p class="muted">Wahi ${m.experts.length} models, ek baar har result se seekh kar (weights badal kar), ek baar bina seekhe (sab barabar). Line = pichle ${pg.window} din ka top-10 hit-rate.</p>
      <div class="grid stats">
        <div class="stat"><div class="s">Seekh kar (learned)</div><div class="v">${pct(pg.learned.hit10?.rate)}</div><div class="s">${cmp(pg.learned)}</div></div>
        <div class="stat"><div class="s">Bina seekhe (equal)</div><div class="v">${pct(pg.equal.hit10?.rate)}</div><div class="s">${cmp(pg.equal)}</div></div>
        <div class="stat"><div class="s">Sabse accha akela model</div><div class="v">${pct(pg.best_expert.hit10_rate)}</div><div class="s">${esc(pg.best_expert.label)}</div></div>
        <div class="stat"><div class="s">Random chance</div><div class="v">10.0%</div><div class="s">avg rank 50.5 · log-loss 4.605</div></div>
      </div>
      <div id="pchart" style="margin-top:12px"></div>
      ${pg.tuning && pg.tuning.length ? `<h3>Self-tuning (meta-learning): tool ne khud chuna kitni tezi se seekhe</h3>
        <p class="muted">η = learning speed (bada = ek result se zyada badlaav), α = bhoolne ki dar (bada = purana jaldi bhoole). 9 settings saath chalti hain, jo sahi nikli uska bharosa badhta hai.</p>
        <div class="row">${pg.tuning.map((t, i) => `<span class="pill ${i === 0 ? "good" : ""}">η=${t.eta}, α=${t.alpha}: ${pct(t.weight)}</span>`).join("")}</div>` : ""}
      ${pg.merge ? `<h3>Calculations ka merge</h3>
        <p class="muted">Saare models ki raay do tareeke se jodi jaati hai: linear (sabki raay ka weighted average) aur geometric (jahan sab models sahmat hon wahan tez). Tool results dekh kar khud tay karta hai kitna kaunsa.</p>
        <div class="row"><span class="pill">Linear: ${pct(pg.merge.linear)}</span><span class="pill">Geometric: ${pct(pg.merge.geometric)}</span></div>` : ""}
    </div>` : "";
  const breakCard = sb ? `
    <div class="card"><h2>"Khud ko todo" test (shuffle)</h2>
      <p class="muted">Engine ko ${sb.shuffles} baar aisi history par chalaya jisme dates ka order ulta-pulta kar diya (har number utni hi baar, par koi time-pattern nahi). Agar asli data par engine shuffled se behtar nahi, to usne koi asli pattern nahi pakda.</p>
      <div class="grid stats">
        <div class="stat"><div class="s">Asli data (${sb.days} din)</div><div class="v">${pct(sb.real_hit10)}</div><div class="s">avg rank ${sb.real_mean_rank.toFixed(1)}</div></div>
        <div class="stat"><div class="s">Shuffled data (average)</div><div class="v">${pct(sb.shuffled_hit10)}</div><div class="s">avg rank ${sb.shuffled_mean_rank.toFixed(1)}</div></div>
        <div class="stat"><div class="s">Asli fayda</div><div class="v">${(sb.edge * 100).toFixed(1)} pts</div><div class="s">${esc(sb.verdict)}</div></div>
      </div>
    </div>` : "";
  $("#tab-learning").innerHTML = progressCard + breakCard + `
    <div class="card"><h2>Self-correction: models ka bharosa (weight) har result ke baad</h2>
      <p class="muted formula">w<sub>i</sub> ← w<sub>i</sub> · P<sub>i</sub>(asli number) → normalize → 99% + 1% barabar baanto (Fixed-Share Hedge)</p>
      <div id="wchart"></div>
    </div>
    <div class="card table-wrap"><h2>${m.experts.length} models (experts)</h2><table>
      <thead><tr><th>Model aur uski math</th><th class="num">Weight</th><th class="num">Avg rank<br><small>(random 50.5)</small></th>
      <th class="num">Top-10 rate<br><small>(random 10%)</small></th><th class="num">Log-loss<br><small>(random 4.605)</small></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  drawWeights(m);
}

const PALETTE = ["#e09c00", "#3b82f6", "#22a55a", "#e5484d", "#9b6cf0", "#14a3b8", "#f06a2e", "#c050a8", "#8b98a9"];

// Plain SVG line chart (no external library, works offline).
function lineChart(dates, series) {
  const entries = Object.entries(series);
  const n = dates.length;
  if (n < 2 || !entries.length) return '<p class="muted">Chart ke liye abhi data kam hai.</p>';
  const W = 1000, H = 320, L = 48, R = 12, T = 12, B = 30;
  const top = Math.max(0.05, ...entries.flatMap(([, d]) => d.filter((v) => v != null)));
  const maxY = Math.min(1, Math.ceil(top * 20) / 20);
  const x = (i) => L + (i * (W - L - R)) / (n - 1);
  const y = (v) => T + (1 - v / maxY) * (H - T - B);
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = (maxY * k) / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/>
          <text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`;
  }
  for (let k = 0; k < 6; k++) {
    const i = Math.round((k * (n - 1)) / 5);
    g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${esc(dates[i].slice(5))}</text>`;
  }
  const paths = entries.map(([label, d], si) => {
    const pts = d.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean);
    return `<polyline points="${pts.join(" ")}" fill="none" stroke="${PALETTE[si % PALETTE.length]}" stroke-width="2"
      vector-effect="non-scaling-stroke"><title>${esc(label)}</title></polyline>`;
  }).join("");
  const legend = entries.map(([label, d], si) => `<span class="lg"><i style="background:${PALETTE[si % PALETTE.length]}"></i>
      ${esc(label)} <b>${pct(d[d.length - 1])}</b></span>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Model weights over time">${g}${paths}</svg>
    <div class="legend">${legend}</div>`;
}

function drawWeights(m) {
  const el = $("#wchart");
  if (el) el.innerHTML = lineChart(m.weights.dates, m.weights.series);
  const pc = $("#pchart");
  if (pc && m.progress) {
    const { dates, ...series } = m.progress.series;
    pc.innerHTML = lineChart(dates, series);
  }
}

// --------------------------------------------------------------- formulas
function renderFormulas() {
  const m = cur();
  const fr = m.formulas;
  if (!m.ready || !fr || !fr.ready) {
    $("#tab-formulas").innerHTML = `<div class="card">${esc(m.reason || (fr && fr.reason) || "")}</div>`;
    return;
  }
  const names = { jodi: "Jodi formule (mod 100)", andar: "Andar formule (mod 10)", bahar: "Bahar formule (mod 10)",
    aryabhata: "Aryabhata kuttaka: (a·pichla + c) mod 100 — 10,000 linear congruences",
    genetic: "Genetic programming — tool ke khud evolve kiye formule (+ − ×, ulta, cut, jod, beejank, jodi)" };
  const blocks = Object.entries(fr.kinds).map(([kind, k]) => `
    <div class="card table-wrap"><h2>${names[kind]}</h2>
      <p class="muted">${k.tested} formule try kiye. Random chance: ${pct(k.chance_rate, 0)}.
      Train = pehle ${fr.train_days} din (jahan formula dhoonda), Test = aakhri ${fr.test_days} din (formula ne kabhi nahi dekha).</p>
      <table><thead><tr><th>Formula</th><th class="num">Train</th><th class="num">Test (unseen)</th><th class="num">p-value</th><th>Verdict</th></tr></thead>
      <tbody>${k.top.map((r) => `<tr>
        <td class="formula">${esc(r.formula)}</td>
        <td class="num">${r.train_hits}/${r.train_n}<br><small class="muted">${pct(r.train_rate)}</small></td>
        <td class="num">${r.test_hits}/${r.test_n}<br><small class="muted">${pct(r.test_rate)}</small></td>
        <td class="num">${pv(r.p_value)}</td>
        <td>${r.significant ? '<span class="pill good">asli pattern</span>' : '<span class="pill">chance jaisa</span>'}</td></tr>`).join("")}
      </tbody></table>
      ${k.next.length ? `<h3>Agle result ke liye (saare data par fit)</h3>${k.next.map((n) => `<div class="formula">${esc(n.formula)} → <b>${kind === "andar" || kind === "bahar" ? n.value : jd(n.value)}</b> <span class="muted">(${n.hits}/${n.n})</span></div>`).join("")}` : ""}
    </div>`).join("");
  $("#tab-formulas").innerHTML = `
    <div class="card"><h2>Tool ke khud ke formule</h2>
      <p>Tool saikdon formule <span class="formula">(c1·U + c2·V + k) mod 100</span> banata hai — U, V = pichla number, doosre markets ka kal ka number, unka ulta, tareekh, din...
      Train data par sabse accha formula hamesha accha dikhega (kyunki itne try kiye). Isliye asli saboot sirf <b>Test (unseen)</b> column hai.</p>
    </div>${blocks}`;
}

// --------------------------------------------------------------- theorems
function renderTheorems() {
  const m = cur();
  if (!m.ready) { $("#tab-theorems").innerHTML = `<div class="card">${esc(m.reason)}</div>`; return; }
  const cards = (m.theorems || []).map((t) => `
    <div class="card">
      <div class="row" style="justify-content:space-between"><h2>${esc(t.id)} · ${esc(t.title)}</h2>
      <span class="pill ${t.pattern === true ? "good" : t.pattern === false ? "" : "info"}">${esc(t.verdict)}</span></div>
      <div>${esc(t.statement)}</div>
      <div class="formula muted">${esc(t.stat)} ${t.p_value != null ? "· " + pv(t.p_value) : ""}</div>
      ${t.note ? `<div class="muted" style="font-size:12px">${esc(t.note)}</div>` : ""}
    </div>`).join("");
  $("#tab-theorems").innerHTML = `<div class="card"><h2>${esc(m.name)}: tool ki statistical findings</h2>
    <p class="muted">Har finding asli data par test hoti hai. "PATTERN MILA" tabhi likha jaata hai jab p-value Bonferroni-corrected limit se chhota ho.</p></div>
    <div class="grid">${cards}</div>`;
}

// ------------------------------------------------------------------ chart
function renderChart() {
  const d = state.dash;
  const keys = Object.keys(d.markets);
  const months = [...new Set(d.results.map((r) => r.date.slice(0, 7)))];
  const sel = state.month && months.includes(state.month) ? state.month : "all";
  const rows = d.results.filter((r) => sel === "all" || r.date.startsWith(sel));
  $("#tab-chart").innerHTML = `
    <div class="card"><div class="row" style="justify-content:space-between">
      <h2>Result chart (${esc(d.history_start)} se aaj tak) — ${d.results.length} din</h2>
      <select id="monthSel"><option value="all">Saare mahine</option>${months.map((mo) => `<option ${mo === sel ? "selected" : ""}>${mo}</option>`).join("")}</select>
    </div>
    ${state.mode === "server" ? '<a href="api/results.csv">CSV download</a>' : ""}</div>
    <div class="card table-wrap"><table>
      <thead><tr><th>Date</th>${keys.map((k) => `<th class="num">${esc(d.markets[k].short)}<br><small class="muted">${esc(d.markets[k].result_time)}</small></th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${esc(r.date)}</td>${keys.map((k) => `<td class="num">${jd(r[k])}</td>`).join("")}</tr>`).join("")
        || `<tr><td colspan="${keys.length + 1}" class="muted">Abhi data nahi.</td></tr>`}</tbody></table></div>`;
  $("#monthSel").addEventListener("change", (e) => { state.month = e.target.value; renderChart(); });
}

// ------------------------------------------------------------------- init
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  state.tab = b.dataset.tab;
  if (state.dash) renderMarketBar();
}));

$("#refresh").addEventListener("click", async () => {
  const btn = $("#refresh");
  btn.disabled = true;
  btn.textContent = "Chal raha hai…";
  try {
    const r = await fetch("api/cycle", { method: "POST" });
    if (!r.ok) alert((await r.json()).detail || "Error");
  } catch (e) { alert("Server se connect nahi hua"); }
  btn.disabled = false;
  btn.textContent = "Abhi Update Karo";
  load();
});

load();
setInterval(load, 5 * 60 * 1000);
