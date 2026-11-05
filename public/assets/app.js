const $ = (selector) => document.querySelector(selector);
const pretty = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : value;
const numeric = (value) => value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
const line = (value) => numeric(value) === null ? "—" : numeric(value).toFixed(1);
const spread = (value) => numeric(value) === null ? "—" : `${numeric(value) > 0 ? "+" : ""}${numeric(value).toFixed(1)}`;
const favoriteName = (marketLine) => numeric(marketLine?.spread_favorite) ? marketLine.favorite_team : null;
function spreadWithTeam(value, team) {
  const formatted = spread(value);
  if (numeric(value) === null || typeof team !== "string" || !team.trim()) return formatted;
  const safeTeam = team.trim().replace(/[&<>"']/g, char => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
  return `${formatted} ${safeTeam}`;
}

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
function animateGameCards(container) {
  container.querySelectorAll(".game-card").forEach(gameCard => {
    gameCard.classList.remove("entering");
    void gameCard.offsetWidth;
    gameCard.classList.add("entering");
  });
}
function animateOnOpen(details) {
  let closing = false;
  details.querySelector("summary").addEventListener("click", event => {
    if (details.open) {
      event.preventDefault();
      if (closing) return;

      const cards = [...details.querySelectorAll(".game-card")]
        .filter(gameCard => !gameCard.closest("details:not([open])"));
      if (!cards.length || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        details.open = false;
        return;
      }

      closing = true;
      const exitOrder = cards.reverse();
      let remaining = exitOrder.length;
      exitOrder.forEach((gameCard, index) => {
        gameCard.classList.remove("entering");
        gameCard.style.setProperty("--exit-delay", `${Math.min(index, 3) * 35}ms`);
        gameCard.addEventListener("animationend", animationEvent => {
          if (animationEvent.target !== gameCard) return;
          gameCard.classList.remove("leaving");
          gameCard.style.removeProperty("--exit-delay");
          remaining -= 1;
          if (remaining === 0) {
            details.open = false;
            closing = false;
          }
        }, {once:true});
        gameCard.classList.add("leaving");
      });
      return;
    }

    if (closing) {
      event.preventDefault();
      return;
    }
    window.setTimeout(() => {
      if (details.open) animateGameCards(details);
    }, 0);
  });
}
function card(game) {
  const template = $("#game-template").content.cloneNode(true);
  const gameCard = template.querySelector(".game-card");
  template.querySelector(".kickoff").textContent = formatStart(game.start);
  template.querySelector(".venue").textContent = pretty(game.venue, game.neutral_site ? "Neutral site" : "");
  template.querySelector(".teams").innerHTML = teamRow(game.away, game.favorite_team) + teamRow(game.home, game.favorite_team);
  const market = game.market || {};
  const opening = market.opening || {};
  const current = market.current || {};
  const modelMargin = numeric(game.projected_favorite_margin);
  const referenceSpread = numeric(opening.spread_favorite);
  template.querySelector(".market-row").innerHTML = `
    <div><span>Reference Spread</span><strong>${spreadWithTeam(opening.spread_favorite, favoriteName(opening))}</strong></div>
    <div><span>Current Spread</span><strong>${spreadWithTeam(current.spread_favorite, favoriteName(current))}</strong></div>
    <div><span>Model Spread</span><strong>${spreadWithTeam(modelMargin === null ? null : -modelMargin, favoriteName(opening))}</strong></div>
    <div><span>Model Total / Market Total</span><strong>${line(game.projected_total)} / ${line(current.total)}</strong></div>`;
  const foot = template.querySelector(".card-foot");
  if (referenceSpread !== null && modelMargin !== null) {
    const edge = modelMargin + referenceSpread;
    if (game.pick) foot.innerHTML = `<span class="pick">MODEL PICK: ${game.pick}</span> · ${Math.abs(edge).toFixed(1)} points versus the reference spread`;
    else foot.textContent = `Pick 'em · margin versus the reference spread: ${spread(edge)} points`;
  } else foot.innerHTML = `<span class="unavailable">Reference spread or model projection unavailable.</span>`;
  if (game.pick && (["win", "loss", "push"].includes(game.ats_result))) {
    foot.classList.add(`outcome-${game.ats_result}`);
  } else if (!game.pick && game.su_result) {
    foot.classList.add("outcome-none");
  }
  gameCard.setAttribute("aria-label", `Open ${game.away.name} at ${game.home.name} game breakdown`);
  gameCard.addEventListener("click", () => openGameDetail(game));
  gameCard.addEventListener("keydown", event => {
    if (event.target !== gameCard || !["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    openGameDetail(game);
  });
  return template;
}

const detailDialog = $("#game-detail");
let selectedDetailGame = null;

const advancedMetrics = [
  {key:"offense_ppa", label:"Offense PPA", short:"Off. PPA"},
  {key:"offense_successRate", label:"Success rate", short:"Off. success"},
  {key:"offense_explosiveness", label:"Explosiveness", short:"Off. explosive"},
  {key:"defense_ppa", label:"PPA allowed", short:"Def. PPA"},
  {key:"defense_successRate", label:"Success rate allowed", short:"Def. success"},
  {key:"defense_explosiveness", label:"Explosiveness allowed", short:"Def. explosive"},
];

function normalizedScore(game, side, metric) {
  return numeric(game.advanced_stats_normalized?.[side]?.[metric.key]);
}

function detailStatHeader(game) {
  return `<div class="detail-stat-legend">
    <span><i class="stat-swatch away"></i>${game.away.name}</span>
    <span><i class="stat-swatch home"></i>${game.home.name}</span>
  </div>`;
}

function detailTeamHeader(game) {
  return [["away", game.away, "AWAY"], ["home", game.home, game.neutral_site ? "HOME · NEUTRAL" : "HOME"]]
    .map(([side, team, label]) => {
      const logo = Number.isInteger(team.logo_id)
        ? `<img class="detail-team-logo" src="assets/logos/${team.logo_id}.png" alt="">`
        : "";
      const pickBadge = team.name?.trim().toLowerCase() === game.pick?.trim().toLowerCase()
        ? `<span class="detail-model-pick">MODEL PICK</span>`
        : "";
      return `<div class="detail-team-card ${side}">
        ${logo}<div class="detail-team-copy"><span>${label}</span><div class="detail-team-title"><strong>${team.name}</strong>${pickBadge}</div></div>
        <div class="detail-team-projection"><b>${line(team.projected_score)}</b><span>PROJ.</span></div>
      </div>`;
    }).join("");
}

function detailBettingInfo(game) {
  const market = game.market || {};
  const opening = market.opening || {};
  const current = market.current || {};
  const modelMargin = numeric(game.projected_favorite_margin);
  const modelMarginLabel = spreadWithTeam(modelMargin === null ? null : -modelMargin, favoriteName(opening));

  return `<div class="detail-betting">
    <div class="detail-market-row">
      <div><span>Reference Spread</span><strong>${spreadWithTeam(opening.spread_favorite, favoriteName(opening))}</strong></div>
      <div><span>Current Spread</span><strong>${spreadWithTeam(current.spread_favorite, favoriteName(current))}</strong></div>
      <div><span>Model Spread</span><strong>${modelMarginLabel}</strong></div>
      <div><span>Model Total / Market Total</span><strong>${line(game.projected_total)} / ${line(current.total)}</strong></div>
    </div>
  </div>`;
}

function renderPointsChart(game) {
  const rows = [
    {key:"points_for", label:"Points for"},
    {key:"points_against", label:"Points against"},
  ];
  const values = rows.flatMap(row => ["away", "home"].map(
    side => numeric(game.pregame_stats?.[side]?.[row.key])
  )).filter(value => value !== null);
  if (!values.length) {
    return `<section class="points-panel"><h3>2026 Averages</h3><p>Scoring averages unavailable.</p></section>`;
  }
  const maximum = Math.ceil(Math.max(...values) / 5) * 5 || 5;
  const groups = rows.map((row, index) => {
    const teamRows = [["away", game.away], ["home", game.home]].map(([side, team]) => {
      const value = numeric(game.pregame_stats?.[side]?.[row.key]);
      return `<span class="points-team-name">${team.name}</span>
        <div class="points-bar-track"><i class="${side}" style="width:${value === null ? 0 : value / maximum * 100}%"></i></div>
        <b class="points-value">${value === null ? "—" : value.toFixed(1)}</b>`;
    }).join("");
    return `<div class="points-group-title${index === 0 ? " first" : ""}"><strong>${row.label}</strong></div>
      <span class="points-axis-spacer"></span>
      <div class="points-axis"><span>0</span><span>${maximum} pts</span></div>
      ${teamRows}`;
  }).join("");
  return `<section class="points-panel">
    <h3>2026 Averages</h3>
    <div class="points-grid">
      ${groups}
    </div>
  </section>`;
}

function renderCoverage(game) {
  const centerX = 220;
  const centerY = 174;
  const radius = 108;
  const angle = index => (-90 + index * 360 / advancedMetrics.length) * Math.PI / 180;
  const point = (index, value, distance = radius) => {
    const radians = angle(index);
    const scaled = distance * value / 100;
    return [centerX + Math.cos(radians) * scaled, centerY + Math.sin(radians) * scaled];
  };
  const polygonPoints = values => values.map((value, index) => {
    const [x, y] = point(index, value ?? 0);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const ring = level => polygonPoints(advancedMetrics.map(() => level));
  const spokes = advancedMetrics.map((metric, index) => {
    const [x, y] = point(index, 100);
    const [labelX, labelY] = point(index, 100, 145);
    const anchor = Math.abs(Math.cos(angle(index))) < 0.25
      ? "middle"
      : Math.cos(angle(index)) > 0 ? "start" : "end";
    return `<line class="radar-axis" x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}"></line>
      <text class="radar-label" x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" text-anchor="${anchor}">${metric.short}</text>`;
  }).join("");
  const scaleMarks = [0, 25, 50, 75, 100].map(level => {
    const y = centerY - radius * level / 100;
    return `<text class="radar-scale-label" x="${centerX + 5}" y="${y + 3}">${level}%</text>`;
  }).join("");
  const awayScores = advancedMetrics.map(metric => normalizedScore(game, "away", metric));
  const homeScores = advancedMetrics.map(metric => normalizedScore(game, "home", metric));
  if (awayScores.every(value => value === null) || homeScores.every(value => value === null)) {
    return `<p class="detail-chart-empty">Week-level advanced stats are unavailable for this matchup.</p>`;
  }
  const dots = (scores, side) => scores.map((score, index) => {
    if (score === null) return "";
    const [x, y] = point(index, score);
    return `<circle class="radar-dot ${side}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3"></circle>`;
  }).join("");
  return `${detailStatHeader(game)}
    <div class="coverage-layout"><div class="coverage-wrap"><svg class="coverage-plot" viewBox="0 0 440 350" role="img" aria-label="Radar coverage plot comparing normalized advanced stats for ${game.away.name} and ${game.home.name}">
      <polygon class="radar-grid" points="${ring(25)}"></polygon><polygon class="radar-grid" points="${ring(50)}"></polygon><polygon class="radar-grid" points="${ring(75)}"></polygon><polygon class="radar-grid" points="${ring(100)}"></polygon>
      ${spokes}
      <polygon class="radar-area away" points="${polygonPoints(awayScores)}"></polygon>
      <polygon class="radar-area home" points="${polygonPoints(homeScores)}"></polygon>
      ${dots(awayScores, "away")}${dots(homeScores, "home")}${scaleMarks}
    </svg></div>${renderPointsChart(game)}</div>`;
}

function renderDetailView() {
  const game = selectedDetailGame;
  if (!game) return;
  $(".detail-teams").innerHTML = detailTeamHeader(game);
  $(".detail-info").innerHTML = detailBettingInfo(game);
  $(".detail-content").innerHTML = renderCoverage(game);
}

function openGameDetail(game) {
  selectedDetailGame = game;
  renderDetailView();
  detailDialog.showModal();
  $(".detail-close").focus();
}

$(".detail-close").addEventListener("click", () => detailDialog.close());
detailDialog.addEventListener("click", event => {
  if (event.target === detailDialog) detailDialog.close();
});

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
    animateOnOpen(section);
    for (const window of day.windows.values()) {
      const windowSection = document.createElement(window.label ? "details" : "div");
      if (window.label) {
        windowSection.className = "game-window";
        windowSection.open = !day.past;
        const subheading = document.createElement("summary");
        subheading.className = "game-window-heading";
        subheading.textContent = window.label;
        windowSection.append(subheading);
        animateOnOpen(windowSection);
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
    animateGameCards($("#game-board"));
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
