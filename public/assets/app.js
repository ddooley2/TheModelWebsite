const $ = (selector) => document.querySelector(selector);
const pretty = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : value;
const numeric = (value) => value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
const line = (value) => numeric(value) === null ? "—" : numeric(value).toFixed(1);
const spread = (value) => numeric(value) === null ? "—" : `${numeric(value) > 0 ? "+" : ""}${numeric(value).toFixed(1)}`;

function showView() {
  const target = location.hash.slice(1) || "board";
  document.querySelectorAll(".view").forEach(view => view.classList.toggle("hidden", view.id !== target));
  document.querySelectorAll("nav a").forEach(link => link.classList.toggle("active", link.dataset.view === target));
}
window.addEventListener("hashchange", showView);
showView();

function teamRow(team, favoriteTeam) {
  const rank = team.rank ? `#${team.rank}` : "";
  const isFavorite = team.name?.trim().toLowerCase() === favoriteTeam?.trim().toLowerCase();
  return `<div class="team${isFavorite ? " favorite" : ""}"><span class="rank">${rank}</span><span><span class="name">${team.name}</span> <span class="record">${pretty(team.record, "")}</span></span><span class="score">${team.projected_score.toFixed(1)}</span><span class="score-label">PROJ.</span></div>`;
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
  template.querySelector(".teams").innerHTML = teamRow(game.away, game.favorite_team) + teamRow(game.home, game.favorite_team);
  const market = game.market || {};
  const opening = market.opening || {};
  const current = market.current || {};
  const modelMargin = numeric(game.projected_favorite_margin);
  const currentSpread = numeric(current.spread_favorite);
  template.querySelector(".market-row").innerHTML = `
    <div><span>Opening favorite spread</span><strong>${spread(opening.spread_favorite)}</strong></div>
    <div><span>Current favorite spread ${pretty(market.provider, "")}</span><strong>${spread(current.spread_favorite)}</strong></div>
    <div><span>Model favorite spread</span><strong>${modelMargin === null ? "—" : spread(-modelMargin)}</strong></div>
    <div><span>Total · model projected / current market</span><strong>${line(game.projected_total)} / ${line(current.total)}</strong></div>`;
  const foot = template.querySelector(".card-foot");
  if (currentSpread !== null && modelMargin !== null) {
    const edge = modelMargin + currentSpread;
    if (game.pick) foot.innerHTML = `<span class="pick">MODEL PICK: ${game.pick}</span> · ${Math.abs(edge).toFixed(1)} points versus the current favorite spread`;
    else foot.textContent = `No published pick · model edge: ${spread(edge)} points (favorite perspective)`;
  } else foot.innerHTML = `<span class="unavailable">Current favorite spread or model projection unavailable.</span>`;
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
