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
  const logo = Number.isInteger(team.logo_id) ? `<img class="team-logo" src="assets/logos/${team.logo_id}.png" alt="" loading="lazy">` : "";
  return `<div class="team${isFavorite ? " favorite" : ""}"><span class="rank">${rank}</span><span class="team-info">${logo}<span class="team-copy"><span class="name">${team.name}</span> <span class="record">${pretty(team.record, "")}</span></span></span><span class="score">${team.projected_score.toFixed(1)}</span><span class="score-label">PROJ.</span></div>`;
}
function formatStart(value) {
  if (!value) return "Time TBD";
  const date = new Date(value);
  return Number.isNaN(date) ? "Time TBD" : new Intl.DateTimeFormat(undefined, {weekday:"short", month:"short", day:"numeric", hour:"numeric", minute:"2-digit"}).format(date);
}
function card(game) {
  const template = $("#game-template").content.cloneNode(true);
  template.querySelector(".kickoff").textContent = formatStart(game.start);
  template.querySelector(".venue").textContent = pretty(game.venue, game.neutral_site ? "Neutral site" : "");
  template.querySelector(".teams").innerHTML = teamRow(game.away, game.favorite_team) + teamRow(game.home, game.favorite_team);
  const market = game.market || {};
  const opening = market.opening || {};
  const current = market.current || {};
  const modelMargin = numeric(game.projected_favorite_margin);
  const currentSpread = numeric(current.spread_favorite);
  template.querySelector(".market-row").innerHTML = `
    <div><span>Opening Spread</span><strong>${spread(opening.spread_favorite)}</strong></div>
    <div><span>Current Spread ${pretty(market.provider, "")}</span><strong>${spread(current.spread_favorite)}</strong></div>
    <div><span>Model Spread</span><strong>${modelMargin === null ? "—" : spread(-modelMargin)}</strong></div>
    <div><span>Model Total / Market Total</span><strong>${line(game.projected_total)} / ${line(current.total)}</strong></div>`;
  const foot = template.querySelector(".card-foot");
  if (currentSpread !== null && modelMargin !== null) {
    const edge = modelMargin + currentSpread;
    if (game.pick) foot.innerHTML = `<span class="pick">MODEL PICK: ${game.pick}</span> · ${Math.abs(edge).toFixed(1)} points versus the current spread`;
    else foot.textContent = `Pick 'em · model edge: ${spread(edge)} points`;
  } else foot.innerHTML = `<span class="unavailable">Current spread or model projection unavailable.</span>`;
  return template;
}
async function loadBoard() {
  try {
    const response = await fetch("data/dashboard.json", {cache:"no-store"});
    if (!response.ok) throw new Error("Dashboard data is not yet generated.");
    const data = await response.json();
    $("#board-title").textContent = `${data.season} · Week ${data.week}`;
    const formatTimestamp = (value) => {
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? null : new Intl.DateTimeFormat(undefined,{dateStyle:"medium",timeStyle:"short"}).format(date);
    };
    const predictionTime = formatTimestamp(data.generated_at);
    const lineTimes = data.games
      .map(game => game.market?.current?.observed_at)
      .filter(value => formatTimestamp(value) !== null);
    const lineTime = lineTimes.sort().at(-1);
    $("#generated-at").textContent = `Predictions generated ${predictionTime}. Lines last updated ${formatTimestamp(lineTime) || "unavailable"}.`;
    const board = $("#game-board"); board.replaceChildren(...data.games.map(card));
  } catch (error) {
    $("#generated-at").textContent = error.message;
  }
}
loadBoard();
