"""Can the goal-scorer predictions predict a game's TOTAL score (both teams)?

Each skater's P(goal) -> Poisson rate lambda = -ln(1-P) (so multi-goal nights
count); sum all skaters on both teams = predicted total goals. Compare to the
actual combined final goals, and test over/under skill.

Part A: OOS skill (train DEV 2020-24, test 2024-26) vs actual totals + a
        synthetic line (own median) to see if ranking games high/low works.
Part B: leak-free O/U vs REAL Vegas totals (2021-22 csv). Retrain scorer GBM
        leaving 2021-22 out, predict it, compare to the posted over_under.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["g_pg", "sh_pg", "satt_pg", "ixg_pg", "hd_pg", "pp_ixg_pg",
            "finish", "toi_pg", "gp", "g_l5", "ixg_l5", "sh_l5", "g_l10",
            "ixg_l10", "is_home", "is_def", "opp_goalie", "opp_def",
            "rate_shrunk", "g_rate_shrunk", "toi_l5", "toi_trend",
            "pp_ixg_l5", "pp_trend"]
ALL = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]


def fit_predict(train_seasons, test_season, df):
    tr = df[df.season.isin(train_seasons)]
    te = df[df.season == test_season].copy()
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0
    ).fit(tr[FEATURES].astype(float), tr["scored"].values)
    te["p"] = gbm.predict_proba(te[FEATURES].astype(float))[:, 1]
    return te


def predicted_totals(te):
    te = te.assign(lam=-np.log(np.clip(1 - te["p"], 1e-9, 1.0)))
    g = te.groupby("game_id").agg(pred_total=("lam", "sum")).reset_index()
    return g.set_index("game_id")["pred_total"]


def load_games(seasons):
    g = pd.concat([pd.read_parquet(f"data/games_{s}.parquet") for s in seasons])
    g["actual_total"] = g["home_goals"] + g["away_goals"]
    return g.set_index("game_id")


def part_a(df):
    print("=" * 60)
    print("PART A: totals skill OOS (train 2020-24, test 2024-26)")
    print("=" * 60)
    pieces = []
    for s in [20242025, 20252026]:
        te = fit_predict([20202021, 20212022, 20222023, 20232024], s, df)
        pt = predicted_totals(te)
        g = load_games([str(s)])
        d = pd.DataFrame({"pred": pt}).join(g["actual_total"], how="inner")
        d["season"] = s
        pieces.append(d)
    d = pd.concat(pieces)

    pred, act = d["pred"], d["actual_total"]
    print(f"\nn={len(d)} games")
    print(f"mean predicted total : {pred.mean():.2f}")
    print(f"mean actual total    : {act.mean():.2f}  "
          f"(raw bias {pred.mean()-act.mean():+.2f})")
    print(f"correlation pred vs actual : {np.corrcoef(pred, act)[0,1]:.3f}")
    print(f"MAE raw          : {(pred-act).abs().mean():.3f}")
    # recenter to actual mean (removes the constant bias the scorer pool has)
    shift = act.mean() - pred.mean()
    predc = pred + shift
    print(f"MAE recentered   : {(predc-act).abs().mean():.3f}")
    print(f"MAE naive (always predict mean) : "
          f"{(act-act.mean()).abs().mean():.3f}")

    # over/under vs our OWN median line: does ranking games high/low work?
    line = predc.median()
    over = act > line
    pick_over = predc > line
    push = act == line
    m = ~push
    acc = (pick_over[m] == over[m]).mean()
    print(f"\nsynthetic O/U @ our median line {line:.2f}: "
          f"pick-side accuracy {acc:.3f} (n={m.sum()}, base over "
          f"{over[m].mean():.3f})")
    # confidence by distance from line
    dist = (predc - line).abs()
    q = pd.qcut(dist, 4, labels=["closest", "Q2", "Q3", "widest"])
    t = pd.DataFrame({"q": q, "correct": (pick_over == over).astype(int)})[m]
    print("  accuracy by |pred - line| quartile:")
    for name, grp in t.groupby("q", observed=True):
        print(f"    {name:<8} n={len(grp):>5}  acc={grp.correct.mean():.3f}")
    return shift  # calibration offset to reuse


def part_b(df):
    print("\n" + "=" * 60)
    print("PART B: O/U vs REAL Vegas totals, 2021-22 (leak-free LOSO)")
    print("=" * 60)
    te = fit_predict([s for s in ALL if s != 20212022], 20212022, df)
    pt = predicted_totals(te)
    g = load_games(["20212022"]).reset_index()

    # vegas lines csv -> match by date + team set
    v = pd.read_csv("sportsbook-nhl-2021-2022.csv")
    v = v.rename(columns={"h__team": "home", "a__team": "away",
                          "over_under": "line"})
    # team abbrev fixes (csv uses TB/NJ etc, games use TBL/NJD...)
    fix = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK"}
    v["home"] = v["home"].replace(fix)
    v["away"] = v["away"].replace(fix)
    v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
    g["date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")

    m = g.merge(v[["date", "home", "away", "line"]], on=["date", "home", "away"],
                how="inner")
    m = m.join(pt, on="game_id")
    m = m.dropna(subset=["line", "pred_total"])
    m["line"] = m["line"].astype(float)
    print(f"matched {len(m)} / {len(g)} games to Vegas lines")

    # recalibrate our total to the real-goal scale using Part A's offset is
    # cross-season; safer: shift to match the VEGAS mean (lines ~ true mean).
    shift = m["line"].mean() - m["pred_total"].mean()
    m["pred_c"] = m["pred_total"] + shift
    print(f"mean line {m['line'].mean():.2f}  "
          f"mean pred(raw) {m['pred_total'].mean():.2f}  "
          f"shift {shift:+.2f}  mean actual {m['actual_total'].mean():.2f}")

    push = m["actual_total"] == m["line"]
    mm = m[~push].copy()
    mm["over_hit"] = mm["actual_total"] > mm["line"]
    mm["pick_over"] = mm["pred_c"] > mm["line"]
    acc = (mm["pick_over"] == mm["over_hit"]).mean()
    print(f"\nbeat-the-line accuracy : {acc:.3f}  (n={len(mm)}, "
          f"pushes {push.sum()}, break-even ~0.524 at -110)")
    print(f"  base over rate {mm['over_hit'].mean():.3f}")

    edge = (mm["pred_c"] - mm["line"]).abs()
    q = pd.qcut(edge, 4, labels=["smallest", "Q2", "Q3", "largest"],
                duplicates="drop")
    t = pd.DataFrame({"q": q,
                      "correct": (mm["pick_over"] == mm["over_hit"]).astype(int),
                      "edge": edge})
    print("  accuracy by |pred - line| quartile:")
    for name, grp in t.groupby("q", observed=True):
        print(f"    {name:<9} n={len(grp):>4}  edge>={grp.edge.min():.2f}  "
              f"acc={grp.correct.mean():.3f}")


def main():
    df = pd.read_parquet("data/scorer_playergames.parquet")
    df["season"] = df["season"].astype(int)
    part_a(df)
    part_b(df)


if __name__ == "__main__":
    main()
