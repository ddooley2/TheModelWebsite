const $ = (selector) => document.querySelector(selector);
const pretty = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : value;
const line = (value) => value === null || value === undefined ? "—" : Number(value).toFixed(1);
const spread = (value) => value === null || value === undefined ? "—" : `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(1)}`;

function showView() {
  const target = location.hash.slice(1) || "board";
  document.querySelectorAll(".view").forEach(view => view.classList.toggle("hidden", view.id !== target));
  document.querySelectorAll("nav a").forEach(link => link.classList.toggle("active", link.dataset.view === target));
}
window.addEventListener("hashchange", showView);
showView();

function teamRow(team) {
  const rank = team.rank ? `#${team.rank}` : "";
  return `<div class="team"><span class="rank">${rank}</span><span><span class="name">${team.name}</span> <span class="record">${pretty(team.record, "")}</span></span><span class="score">${team.projected_score.toFixed(1)}</span><span class="score-label">PROJ.</span></div>`;
}
function formatStart(value) {
  if (!value) return "Time TBD";
  const date = new Date(value);
  return Number.isNaN(date) ? "Time TBD" : new Intl.DateTimeFormat(undefined, {weekday:"short", month:"short", day:"numeric", hour:"numeric", minute:"2-digit"}).format(date);
}
function card(game) {
  const template = $("#game-template").content.cloneNode(true);
  template.querySelector(".kickoff").textContent = formatStart(game.start);
  template.querySelector(".broadcast").textContent = pretty(game.network, "Network TBD");
  template.querySelector(".venue").textContent = pretty(game.venue, game.neutral_site ? "Neutral site" : "");
  template.querySelector(".teams").innerHTML = teamRow(game.away) + teamRow(game.home);
  const opening = game.market.opening || {};
  const current = game.market.current || {};
  template.querySelector(".market-row").innerHTML = `
    <div><span>Open (home)</span><strong>${spread(opening.spread)}</strong></div>
    <div><span>Current ${pretty(game.market.provider, "")}</span><strong>${spread(current.spread)}</strong></div>
    <div><span>Model / total</span><strong>${spread(-game.projected_home_margin)} / ${line(game.projected_total)}</strong></div>`;
  const foot = template.querySelector(".card-foot");
  if (game.pick) foot.innerHTML = `<span class="pick">MODEL PICK: ${game.pick}</span> · ${Math.abs(game.model_edge_home).toFixed(1)} points versus the current home spread`;
  else if (current.spread !== undefined && current.spread !== null) foot.textContent = `No published pick · model edge: ${line(game.model_edge_home)} points (home perspective)`;
  else foot.innerHTML = `<span class="unavailable">Current CFBD market line unavailable.</span>`;
  return template;
}
async function loadBoard() {
  try {
    const response = await fetch("data/dashboard.json", {cache:"no-store"});
    if (!response.ok) throw new Error("Dashboard data is not yet generated.");
    const data = await response.json();
    $("#board-title").textContent = `${data.season} · Week ${data.week}`;
    $("#generated-at").textContent = `Generated ${new Intl.DateTimeFormat(undefined,{dateStyle:"medium",timeStyle:"short"}).format(new Date(data.generated_at))} · ${data.market_data_status}`;
    const board = $("#game-board"); board.replaceChildren(...data.games.map(card));
  } catch (error) {
    $("#generated-at").textContent = error.message;
  }
}
async function loadDiagnostics() {
  try {
    const response = await fetch("data/diagnostics.json", {cache:"no-store"});
    if (!response.ok) return;
    const data = await response.json();
    const winner = data.winner?.winner || "pending";
    const mae = data.mean_score_mae?.[winner];
    $("#diagnostic-summary").innerHTML = `<strong>${data.games.toLocaleString()}</strong> completed FBS-vs-FBS games · <strong>${data.features.length}</strong> retained inputs · selected market-free candidate: <strong>${winner}</strong>${mae ? ` · mean walk-forward score MAE: <strong>${mae.toFixed(3)}</strong>` : ""}`;
  } catch (_) {}
}
loadBoard(); loadDiagnostics();
