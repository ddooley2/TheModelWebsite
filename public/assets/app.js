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
  return Number.isNaN(date.getTime()) ? "Time TBD" : new Intl.DateTimeFormat(undefined, {weekday:"short", month:"short", day:"numeric", hour:"numeric", minute:"2-digit", timeZone:"America/New_York", timeZoneName:"short"}).format(date);
}
const etDate = date => new Intl.DateTimeFormat("en-CA", {timeZone:"America/New_York"}).format(date);
function gameGroup(game) {
  const date = new Date(game.start);
  if (!game.start || Number.isNaN(date.getTime())) {
    return {key:"tbd", label:"Kickoff time TBD", dayLabel:"Kickoff time TBD", windowLabel:null};
  }
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone:"America/New_York",
    weekday:"long",
    year:"numeric",
    month:"long",
    day:"numeric",
    hour:"2-digit",
    hourCycle:"h23",
  }).formatToParts(date).map(part => [part.type, part.value]));
  const dateKey = `${parts.year}-${parts.month}-${parts.day}`;
  const past = etDate(date) < etDate(new Date());
  const dateLabel = `${parts.weekday}, ${parts.month} ${parts.day}`;
  if (parts.weekday !== "Saturday") {
    return {key:dateKey, label:dateLabel, dayLabel:dateLabel, windowLabel:null, past};
  }

  const hour = Number(parts.hour);
  const window = hour < 14
    ? ["noon", "Noon Window"]
    : hour < 18
      ? ["afternoon", "3:30 PM Window"]
      : hour < 21
        ? ["evening", "Primetime Window"]
        : ["late", "Late Window"];
  return {
    key:dateKey,
    label:dateLabel,
    dayLabel:dateLabel,
    past,
    windowKey:window[0],
    windowLabel:window[1],
  };
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
  const referenceSpread = numeric(opening.spread_favorite);
  template.querySelector(".market-row").innerHTML = `
    <div><span>Reference Spread</span><strong>${spread(opening.spread_favorite)}</strong></div>
    <div><span>Current Spread</span><strong>${spread(current.spread_favorite)}</strong></div>
    <div><span>Model Spread</span><strong>${modelMargin === null ? "—" : spread(-modelMargin)}</strong></div>
    <div><span>Model Total / Market Total</span><strong>${line(game.projected_total)} / ${line(current.total)}</strong></div>`;
  const foot = template.querySelector(".card-foot");
  if (referenceSpread !== null && modelMargin !== null) {
    const edge = modelMargin + referenceSpread;
    if (game.pick) foot.innerHTML = `<span class="pick">MODEL PICK: ${game.pick}</span> · ${Math.abs(edge).toFixed(1)} points versus the reference spread`;
    else foot.textContent = `Pick 'em · margin versus the reference spread: ${spread(edge)} points`;
  } else foot.innerHTML = `<span class="unavailable">Reference spread or model projection unavailable.</span>`;
  if (game.pick && (game.ats_result === "win" || game.ats_result === "loss")) {
    foot.classList.add(`outcome-${game.ats_result}`);
  } else if (!game.pick && game.su_result) {
    foot.classList.add("outcome-none");
  }
  return template;
}
function renderGameBoard(games, filter = "all") {
  const top25Only = filter === "top25";
  const visibleGames = top25Only
    ? games.filter(game => [game.home.rank, game.away.rank].some(rank => {
      const value = numeric(rank);
      return value !== null && value >= 1 && value <= 25;
    }))
    : games;
  const sortedGames = [...visibleGames].sort((a, b) => {
    const startA = a.start ? new Date(a.start).getTime() : NaN;
    const startB = b.start ? new Date(b.start).getTime() : NaN;
    const validA = Number.isNaN(startA) ? Infinity : startA;
    const validB = Number.isNaN(startB) ? Infinity : startB;
    return validA - validB;
  });
  const groups = new Map();
  for (const game of sortedGames) {
    const group = gameGroup(game);
    if (!groups.has(group.key)) {
      groups.set(group.key, {
        label:group.label,
        dayLabel:group.dayLabel,
        past:group.past,
        windows:new Map(),
      });
    }
    const day = groups.get(group.key);
    const windowKey = group.windowKey || "all";
    if (!day.windows.has(windowKey)) {
      day.windows.set(windowKey, {label:group.windowLabel, games:[]});
    }
    day.windows.get(windowKey).games.push(game);
  }

  const sections = [];
  for (const day of groups.values()) {
    const section = document.createElement("details");
    section.className = "game-day";
    section.open = !day.past;
    const heading = document.createElement("summary");
    heading.className = "game-day-heading";
    heading.textContent = day.dayLabel;
    section.append(heading);
    for (const window of day.windows.values()) {
      const windowSection = document.createElement(window.label ? "details" : "div");
      if (window.label) {
        windowSection.className = "game-window";
        windowSection.open = !day.past;
        const subheading = document.createElement("summary");
        subheading.className = "game-window-heading";
        subheading.textContent = window.label;
        windowSection.append(subheading);
      }
      const cards = document.createElement("div");
      cards.className = "game-day-games";
      cards.append(...window.games.map(card));
      windowSection.append(cards);
      section.append(windowSection);
    }
    sections.push(section);
  }
  if (top25Only && sections.length === 0) {
    const empty = document.createElement("p");
    empty.className = "top25-empty";
    empty.textContent = "No games with a Top 25 team.";
    sections.push(empty);
  }
  $("#game-board").replaceChildren(...sections);
}
function performanceSummary(week, current = false) {
  const wins = numeric(week.wins) ?? 0;
  const losses = numeric(week.losses) ?? 0;
  const pushes = numeric(week.pushes) ?? 0;
  const atsPct = numeric(week.ats_pct);
  const suWins = numeric(week.su_wins) ?? 0;
  const suLosses = numeric(week.su_losses) ?? 0;
  const suPct = numeric(week.su_pct);
  const pctLabel = value => value === null ? "—" : `${(value <= 1 ? value * 100 : value).toFixed(1)}%`;
  return `<article class="performance-week${current ? " current-week" : " past-week"}">
    <div class="performance-week-heading"><h2>Week ${week.week}</h2>${current ? '<span class="current-label">CURRENT WEEK</span>' : ""}</div>
    <div class="performance-metrics">
      <div class="performance-metric"><span>ATS · W–L–P</span><strong>${wins}–${losses}–${pushes}</strong><small>${pctLabel(atsPct)} win rate</small></div>
      <div class="performance-metric"><span>STRAIGHT UP · W–L</span><strong>${suWins}–${suLosses}</strong><small>${pctLabel(suPct)} win rate</small></div>
    </div>
  </article>`;
}
function renderPerformance(performance) {
  const container = $("#performance-content");
  const weeks = Array.isArray(performance?.weeks)
    ? performance.weeks.filter(week => week && numeric(week.week) !== null && numeric(week.week) >= 9)
    : [];
  if (!weeks.length) {
    container.innerHTML = '<p class="performance-empty">Performance results are not available yet.</p>';
    return;
  }

  const requestedCurrentWeek = numeric(performance.current_week);
  const currentWeek = requestedCurrentWeek !== null && requestedCurrentWeek >= 9
    ? requestedCurrentWeek
    : Math.max(...weeks.map(week => numeric(week.week)));
  const current = weeks.find(week => numeric(week.week) === currentWeek);
  const past = weeks
    .filter(week => numeric(week.week) !== currentWeek)
    .sort((a, b) => numeric(b.week) - numeric(a.week));
  const currentCard = current
    ? performanceSummary(current, true)
    : `<article class="performance-week current-week"><div class="performance-week-heading"><h2>Week ${currentWeek}</h2><span class="current-label">CURRENT WEEK</span></div><p class="performance-empty-message">Results are not available yet.</p></article>`;
  container.innerHTML = `${currentCard}
    ${past.length ? `<div class="past-weeks">${past.map(week => performanceSummary(week)).join("")}</div>` : ""}`;
}
let boardGames = [];
let boardFilter = "all";
document.querySelectorAll(".board-filter").forEach(button => {
  button.addEventListener("click", () => {
    boardFilter = button.dataset.filter;
    document.querySelectorAll(".board-filter").forEach(filterButton => {
      const active = filterButton === button;
      filterButton.classList.toggle("active", active);
      filterButton.setAttribute("aria-pressed", String(active));
    });
    renderGameBoard(boardGames, boardFilter);
  });
});
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
    boardGames = data.games;
    renderGameBoard(boardGames, boardFilter);
    renderPerformance(data.performance);
  } catch (error) {
    $("#generated-at").textContent = error.message;
    renderPerformance(null);
  }
}
loadBoard();
