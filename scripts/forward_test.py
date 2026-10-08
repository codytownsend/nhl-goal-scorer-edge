"""Live forward test — the only verdict the spec really trusts.

  predict : pull the upcoming slate, predict each game from current ratings,
            (optionally) attach live moneylines, append to a dated ledger.
  grade   : fill finished games' results, report running accuracy / ROI.

    python -m scripts.forward_test predict [--odds]
    python -m scripts.forward_test grade

IMPORTANT honesty notes:
  * Ratings come from the most recent seasons you have BUILT. For a real
    in-season test, rebuild events for the live season regularly so ratings are
    current (otherwise teams are rated on stale/again-missing rosters, and new
    franchises like UTA won't be rated).
  * The NHL API does NOT publish confirmed starting goalies pregame. This uses a
    team-quality + home-ice prediction; supply day-of starters via an external
    feed / manual override to capture the goalie edge (the real orthogonal signal).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nhlsit import games as G
from nhlsit import market, predict, rating, schedule

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "forward_ledger.csv"
SEASONS = ["20202021", "20212022", "20222023", "20232024"]


def _load_events():
    frames = []
    for s in SEASONS:
        fp = ROOT / "data" / f"events_{s}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["season"] = str(s)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _current_model():
    events = _load_events()
    gbs = {s: G.load_or_build(s) for s in SEASONS}
    state = rating.current_state(events, gbs, SEASONS)
    preds, _ = rating.walk_forward(events, gbs, SEASONS)
    mm = predict.MarginMap().fit(preds["pred_xg_margin"], preds["home_margin"])
    return state, mm


def do_predict(attach_odds: bool):
    state, mm = _current_model()
    ratings, home_adv = state["ratings"], state["home_adv"]
    up = schedule.upcoming_games()
    if up.empty:
        print("no upcoming games in the current schedule week.")
        return

    odds = None
    if attach_odds:
        try:
            odds = market.fetch_live_odds()
        except SystemExit as e:
            print(f"(odds skipped: {e})")

    rows = []
    for g in up.itertuples(index=False):
        rh, ra = ratings.get(g.home), ratings.get(g.away)
        if rh is None or ra is None:
            print(f"  skip {g.away}@{g.home}: no rating (stale/new team) — "
                  "rebuild current-season events")
            continue
        xgm = home_adv + rh - ra
        p_home = float(mm.p_home_win(xgm))
        row = {"date": g.date, "game_id": g.game_id, "home": g.home, "away": g.away,
               "p_home": round(p_home, 3),
               "pick": g.home if p_home >= 0.5 else g.away,
               "pred_margin": round(float(mm.margin(xgm)), 2),
               "home_ml": None, "away_ml": None, "result": None, "correct": None}
        if odds is not None:
            o = odds[(odds.home == g.home) & (odds.away == g.away)]
            if len(o):
                row["home_ml"] = float(o.iloc[0]["home_ml"])
                row["away_ml"] = float(o.iloc[0]["away_ml"])
        rows.append(row)

    new = pd.DataFrame(rows)
    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        new = pd.concat([old, new]).drop_duplicates("game_id", keep="last")
    new.to_csv(LEDGER, index=False)
    print(f"logged {len(rows)} predictions -> {LEDGER}")
    print(new[["date", "away", "home", "p_home", "pick"]].tail(len(rows)).to_string(index=False))


def do_grade():
    if not LEDGER.exists():
        raise SystemExit("no ledger yet; run `predict` first.")
    led = pd.read_csv(LEDGER)
    pending = led[led["result"].isna()]
    for date in sorted(pending["date"].unique()):
        res = schedule.results_for_date(date)
        for i, r in led[led["date"] == date].iterrows():
            m = res[res["game_id"] == r["game_id"]]
            if m.empty or pd.isna(m.iloc[0]["home_goals"]):
                continue
            hw = int(m.iloc[0]["home_goals"] > m.iloc[0]["away_goals"])
            led.at[i, "result"] = "home" if hw else "away"
            led.at[i, "correct"] = int((r["pick"] == r["home"]) == bool(hw))
    led.to_csv(LEDGER, index=False)
    graded = led[led["correct"].notna()]
    if len(graded):
        print(f"graded {len(graded)} games | accuracy {graded['correct'].mean():.3f}")
        if graded["home_ml"].notna().any():
            _roi(graded)
    else:
        print("no finished games to grade yet.")


def _roi(graded):
    prof = []
    for r in graded.itertuples(index=False):
        if pd.isna(r.home_ml):
            continue
        ml = r.home_ml if r.pick == r.home else r.away_ml
        dec = market.american_to_decimal(ml)
        prof.append((dec - 1) if r.correct else -1)
    if prof:
        print(f"flat-stake ROI on {len(prof)} priced picks: "
              f"{sum(prof)/len(prof):+.2%} ({sum(prof):+.1f}u)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "grade"])
    ap.add_argument("--odds", action="store_true", help="attach live moneylines")
    args = ap.parse_args()
    if args.cmd == "predict":
        do_predict(args.odds)
    else:
        do_grade()


if __name__ == "__main__":
    main()
