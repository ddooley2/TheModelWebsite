const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadApp() {
  const elements = {};
  const element = () => ({
    classList: {add() {}, toggle() {}},
    addEventListener() {},
    setAttribute() {},
  });
  const template = {
    querySelector(selector) {
      return elements[selector] ||= element();
    },
  };
  const document = {
    querySelector(selector) {
      if (selector === "#game-template") return {content: {cloneNode: () => template}};
      return element();
    },
    querySelectorAll: () => [],
  };
  const context = vm.createContext({
    document,
    window: {addEventListener() {}},
    location: {hash: ""},
    fetch: () => new Promise(() => {}),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../public/assets/app.js"), "utf8"), context);
  return {context, elements};
}

function spreadLabels(html) {
  return [...html.matchAll(/<span>(Reference Spread|Current Spread|Model Spread)<\/span><strong>(.*?)<\/strong>/g)]
    .map(match => match[2]);
}

function checkViews(game, expected) {
  const {context, elements} = loadApp();
  context.game = {
    home: {name: "Alpha", projected_score: 24},
    away: {name: "Beta", projected_score: 21},
    ...game,
  };
  vm.runInContext("card(game)", context);
  assert.deepEqual(spreadLabels(elements[".market-row"].innerHTML), expected);
  assert.deepEqual(spreadLabels(vm.runInContext("detailBettingInfo(game)", context)), expected);
}

test("both views label each spread using its own favorite perspective", () => {
  checkViews({
    market: {
      opening: {spread_favorite: -3.5, favorite_team: "Alpha"},
      current: {spread_favorite: -2, favorite_team: "Beta"},
    },
    projected_favorite_margin: -1.5,
  }, ["-3.5 Alpha", "-2.0 Beta", "+1.5 Alpha"]);
});

test("missing lines and teams do not create a favorite", () => {
  checkViews({
    market: {
      opening: {spread_favorite: -3, favorite_team: null},
      current: {spread_favorite: null, favorite_team: "Beta"},
    },
    projected_favorite_margin: 4,
  }, ["-3.0", "—", "-4.0"]);
  checkViews({
    market: {opening: {favorite_team: "Alpha"}, current: {spread_favorite: -2}},
    projected_favorite_margin: null,
  }, ["—", "-2.0", "—"]);
});

test("pick'em lines do not invent favorites, while a tied model keeps its reference team", () => {
  checkViews({
    market: {
      opening: {spread_favorite: 0, favorite_team: "Alpha"},
      current: {spread_favorite: 0, favorite_team: "Beta"},
    },
    projected_favorite_margin: 0,
  }, ["0.0", "0.0", "0.0"]);
  checkViews({
    market: {opening: {spread_favorite: -1, favorite_team: "A & B"}},
    projected_favorite_margin: 0,
  }, ["-1.0 A &amp; B", "—", "0.0 A &amp; B"]);
});
