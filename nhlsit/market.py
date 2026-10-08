"""Market (moneyline) comparison — benchmark ONLY, never a model input.

The model never sees odds. Here we compare its predictions to the book:
straight-up accuracy head-to-head, who's right on disagreements, and flat-stake
moneyline ROI on the model's +EV side with correct (juiced) break-even.

Supply historical odds as a CSV with columns: date, home, away, home_ml, away_ml
(American odds, e.g. -150 / +130). Team names are normalized to 3-letter codes.
Live odds for upcoming games can be pulled from The Odds API (free tier) with
`fetch_live_odds(api_key)`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Map common full / city names to NHL 3-letter codes (odds feeds vary).
NAME2ABBR = {
    "anaheim": "ANA", "ducks": "ANA", "boston": "BOS", "bruins": "BOS",
    "buffalo": "BUF", "sabres": "BUF", "carolina": "CAR", "hurricanes": "CAR",
    "columbus": "CBJ", "blue jackets": "CBJ", "calgary": "CGY", "flames": "CGY",
    "chicago": "CHI", "blackhawks": "CHI", "colorado": "COL", "avalanche": "COL",
    "dallas": "DAL", "stars": "DAL", "detroit": "DET", "red wings": "DET",
    "edmonton": "EDM", "oilers": "EDM", "florida": "FLA", "panthers": "FLA",
    "los angeles": "LAK", "kings": "LAK", "minnesota": "MIN", "wild": "MIN",
    "montreal": "MTL", "montréal": "MTL", "canadiens": "MTL", "new jersey": "NJD",
    "devils": "NJD", "nashville": "NSH", "predators": "NSH", "ny islanders": "NYI",
    "islanders": "NYI", "ny rangers": "NYR", "rangers": "NYR", "ottawa": "OTT",
    "senators": "OTT", "philadelphia": "PHI", "flyers": "PHI", "pittsburgh": "PIT",
    "penguins": "PIT", "seattle": "SEA", "kraken": "SEA", "san jose": "SJS",
    "sharks": "SJS", "st louis": "STL", "st. louis": "STL", "blues": "STL",
    "tampa bay": "TBL", "lightning": "TBL", "toronto": "TOR", "maple leafs": "TOR",
    "utah": "UTA", "arizona": "ARI", "coyotes": "ARI", "vancouver": "VAN",
    "canucks": "VAN", "vegas": "VGK", "golden knights": "VGK", "winnipeg": "WPG",
    "jets": "WPG", "washington": "WSH", "capitals": "WSH",
    # short scoreboard codes used by some feeds
    "la": "LAK", "nj": "NJD", "sj": "SJS", "tb": "TBL", "was": "WSH",
    "cls": "CBJ", "wsh": "WSH", "tbl": "TBL", "lak": "LAK", "njd": "NJD",
    "sjs": "SJS",
}
ABBREVS = set(NAME2ABBR.values())


def to_abbrev(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    if s.upper() in ABBREVS:
        return s.upper()
    key = s.lower()
    if key in NAME2ABBR:
        return NAME2ABBR[key]
    for token, ab in NAME2ABBR.items():          # substring fallback
        if token in key:
            return ab
    return s.upper()


def american_to_prob(ml: float) -> float:
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else (-ml) / (-ml + 100.0)


def american_to_decimal(ml: float) -> float:
    ml = float(ml)
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / (-ml))


def load_odds_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["date", "home", "away", "home_ml", "away_ml"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"odds CSV missing columns: {missing}. Have {list(df.columns)}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[cols["date"]]).dt.strftime("%Y-%m-%d"),
        "home": df[cols["home"]].map(to_abbrev),
        "away": df[cols["away"]].map(to_abbrev),
        "home_ml": df[cols["home_ml"]].astype(float),
        "away_ml": df[cols["away_ml"]].astype(float),
    })
    ph, pa = out["home_ml"].map(american_to_prob), out["away_ml"].map(american_to_prob)
    out["mkt_home_raw"], out["mkt_away_raw"] = ph, pa
    out["mkt_home_devig"] = ph / (ph + pa)        # vig-free market P(home win)
    return out


def prob_to_american(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def load_favorite_odds_csv(path: str, hold: float = 0.045) -> pd.DataFrame:
    """Loader for feeds that list only the favorite's moneyline + which side.

    Expects columns: date, a__team, h__team, moneyline (favorite's, <=0),
    fav in {Home,Away,Even}. The underdog price is RECONSTRUCTED by assuming a
    two-way hold (default 4.5%): p_dog_raw = (1+hold) - p_fav_raw. Approximate —
    fine for a rough ROI, exact for straight-up accuracy (which uses `fav`).
    """
    df = pd.read_csv(path)
    rows = []
    for r in df.itertuples(index=False):
        home, away = to_abbrev(r.h__team), to_abbrev(r.a__team)
        ml = float(r.moneyline)
        if r.fav == "Even" or ml == 0:
            ph_raw = pa_raw = (1 + hold) / 2
        else:
            p_fav = american_to_prob(ml)
            p_dog = max((1 + hold) - p_fav, 1e-3)
            if r.fav == "Home":
                ph_raw, pa_raw = p_fav, p_dog
            else:
                ph_raw, pa_raw = p_dog, p_fav
        rows.append({
            "date": pd.to_datetime(r.date).strftime("%Y-%m-%d"),
            "home": home, "away": away,
            "home_ml": prob_to_american(ph_raw), "away_ml": prob_to_american(pa_raw),
            "mkt_home_raw": ph_raw, "mkt_away_raw": pa_raw,
            "mkt_home_devig": ph_raw / (ph_raw + pa_raw),
            "fav": r.fav,
        })
    return pd.DataFrame(rows)


def load_twosided_xlsx(path: str) -> pd.DataFrame:
    """Loader for clean two-sided closing-line files (ESPN BET export):
    columns game_time, home_team, away_team, home_ml, away_ml, spread, total,
    home_sp, away_sp, over_odds, under_odds. Returns a tidy odds frame with
    de-vigged market P(home win) plus the puck-line / totals prices.
    """
    df = pd.read_excel(path)
    # game_time is UTC; NHL's game date is the US/Eastern calendar day, so
    # convert before taking the date or late West-coast games land a day off.
    et = pd.to_datetime(df["game_time"], utc=True).dt.tz_convert("America/New_York")
    out = pd.DataFrame({
        "date": et.dt.strftime("%Y-%m-%d"),
        "home": df["home_team"].map(to_abbrev),
        "away": df["away_team"].map(to_abbrev),
        "home_ml": df["home_ml"].astype(float),
        "away_ml": df["away_ml"].astype(float),
        "total_line": df["total"].astype(float),
        "over_odds": df["over_odds"].astype(float),
        "under_odds": df["under_odds"].astype(float),
        "home_goals": df["home_goals"], "away_goals": df["away_goals"],
    })
    ph, pa = out["home_ml"].map(american_to_prob), out["away_ml"].map(american_to_prob)
    out["mkt_home_raw"], out["mkt_away_raw"] = ph, pa
    out["mkt_home_devig"] = ph / (ph + pa)
    return out


def evaluate_vs_market(preds: pd.DataFrame, odds: pd.DataFrame) -> dict:
    """preds must have date, home, away, p_home, home_win. Returns summary +
    a bet ledger for the model's +EV moneyline plays (flat 1u stakes)."""
    m = preds.merge(odds, on=["date", "home", "away"], how="inner")
    if m.empty:
        raise ValueError("no games matched between predictions and odds "
                         "(check date format YYYY-MM-DD and team codes)")
    m["mkt_pick_home"] = (m["mkt_home_devig"] >= 0.5).astype(int)
    m["model_pick_home"] = (m["p_home"] >= 0.5).astype(int)
    m["mkt_correct"] = (m["mkt_pick_home"] == m["home_win"]).astype(int)
    m["model_correct"] = (m["model_pick_home"] == m["home_win"]).astype(int)
    disagree = m[m["mkt_pick_home"] != m["model_pick_home"]]

    # +EV moneyline bets: side where model prob > raw (juiced) implied prob.
    ledger = []
    for r in m.itertuples(index=False):
        for side, p_model, p_raw, ml in (
            ("home", r.p_home, r.mkt_home_raw, r.home_ml),
            ("away", 1 - r.p_home, r.mkt_away_raw, r.away_ml),
        ):
            if p_model > p_raw:                    # model sees value
                won = (r.home_win == 1) if side == "home" else (r.home_win == 0)
                dec = american_to_decimal(ml)
                profit = (dec - 1.0) if won else -1.0
                ledger.append({"date": r.date, "home": r.home, "away": r.away,
                               "side": side, "ml": ml, "p_model": p_model,
                               "p_raw": p_raw, "won": int(won), "profit": profit})
    led = pd.DataFrame(ledger)
    roi = led["profit"].sum() / len(led) if len(led) else float("nan")

    return {
        "n_games": len(m),
        "model_acc": m["model_correct"].mean(),
        "market_acc": m["mkt_correct"].mean(),
        "n_disagree": len(disagree),
        "model_acc_on_disagree": disagree["model_correct"].mean() if len(disagree) else float("nan"),
        "n_bets": len(led),
        "bet_hit_rate": led["won"].mean() if len(led) else float("nan"),
        "roi": roi,
        "units": led["profit"].sum() if len(led) else 0.0,
        "ledger": led,
        "merged": m,
    }


def fetch_live_odds(api_key: str | None = None, regions="us") -> pd.DataFrame:
    """Upcoming-game moneylines from The Odds API (the-odds-api.com, free tier).
    Set THE_ODDS_API_KEY env var or pass api_key. For the live forward test."""
    import requests
    api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise SystemExit("Set THE_ODDS_API_KEY (free at the-odds-api.com).")
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    r = requests.get(url, params={"apiKey": api_key, "regions": regions,
                                  "markets": "h2h", "oddsFormat": "american"},
                     timeout=30)
    r.raise_for_status()
    rows = []
    for g in r.json():
        home, away = to_abbrev(g["home_team"]), to_abbrev(g["away_team"])
        prices = {}
        for bk in g.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                for oc in mk["outcomes"]:
                    prices.setdefault(to_abbrev(oc["name"]), []).append(oc["price"])
        if home in prices and away in prices:
            rows.append({"date": g["commence_time"][:10], "home": home, "away": away,
                         "home_ml": np.median(prices[home]),
                         "away_ml": np.median(prices[away])})
    return pd.DataFrame(rows)


# ---- player props: anytime goal scorer (single cached pull) --------------

PROPS_BASE = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"


def _parse_scorer_props(blob: dict) -> pd.DataFrame:
    """Flatten cached props JSON -> one row per (game, player) with median odds."""
    import numpy as np
    rows = []
    for ev in blob.get("events", []):
        home, away = to_abbrev(ev.get("home_team", "")), to_abbrev(ev.get("away_team", ""))
        prices = {}  # player -> [american odds across books]
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "player_goal_scorer_anytime":
                    continue
                for oc in mk.get("outcomes", []):
                    # anytime-scorer: description=player, name often 'Yes'
                    if str(oc.get("name", "")).lower() in ("no",):
                        continue
                    player = oc.get("description") or oc.get("name")
                    prices.setdefault(player, []).append(float(oc["price"]))
        for player, pr in prices.items():
            rows.append({"game": f"{away}@{home}", "away": away, "home": home,
                         "player": player, "odds": float(np.median(pr))})
    return pd.DataFrame(rows)


def _event_key(e: dict) -> str:
    """'AWAY@HOME' key from an event's team names (works for events list + odds)."""
    return f"{to_abbrev(e.get('away_team', ''))}@{to_abbrev(e.get('home_team', ''))}"


def _resolve_key(api_key):
    import os
    from pathlib import Path
    api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        key_fp = Path(__file__).resolve().parent.parent / ".oddskey"
        if key_fp.exists():
            api_key = key_fp.read_text().strip()
    if not api_key:
        raise SystemExit("Set THE_ODDS_API_KEY or write the key to ./.oddskey")
    return api_key


def fetch_scorer_props(date: str, api_key: str | None = None, region: str = "us",
                       use_cache: bool = True, only_games=None):
    """Incrementally pull anytime-goal-scorer props for `date`.

    Lists the slate (the /events endpoint is free) and fetches odds ONLY for games
    not already in the cache -- optionally restricted to `only_games` (a set of
    'AWAY@HOME' keys). So re-running fills newly-posted games for the cost of just
    those games, and you can target a single missing game. Caches to
    data/raw/props_<date>.json. Returns (df, meta); meta has requests_remaining
    and 'fetched' (how many games this call pulled).
    """
    import json
    import requests
    from pathlib import Path

    cache_fp = Path(__file__).resolve().parent.parent / "data" / "raw" / f"props_{date}.json"
    blob = (json.loads(cache_fp.read_text()) if cache_fp.exists()
            else {"date": date, "events": [], "_meta": {}})
    have = {_event_key(e) for e in blob["events"]}

    want = set(only_games) if only_games is not None else None
    if use_cache and want is not None and want.issubset(have):
        return _parse_scorer_props(blob), {**blob.get("_meta", {}), "fetched": 0}

    api_key = _resolve_key(api_key)
    evr = requests.get(f"{PROPS_BASE}/events",
                       params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    evr.raise_for_status()
    events = [e for e in evr.json() if e["commence_time"][:10] == date]

    meta, fetched = blob.get("_meta", {}), 0
    for e in events:
        k = _event_key(e)
        if k in have:
            continue
        if want is not None and k not in want:
            continue
        r = requests.get(f"{PROPS_BASE}/events/{e['id']}/odds",
                         params={"apiKey": api_key, "regions": region,
                                 "markets": "player_goal_scorer_anytime",
                                 "oddsFormat": "american"}, timeout=30)
        r.raise_for_status()
        meta = {"requests_remaining": r.headers.get("x-requests-remaining"),
                "requests_used": r.headers.get("x-requests-used")}
        blob["events"].append(r.json())
        have.add(k)
        fetched += 1

    blob["_meta"] = meta
    cache_fp.write_text(json.dumps(blob))
    return _parse_scorer_props(blob), {**meta, "fetched": fetched}
