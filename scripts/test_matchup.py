"""Direction A: do style-matchup INTERACTIONS add predictive signal?

Base model assumes transitive team quality. Here we build each team's
walk-forward style tendencies (rush / rebound / high-danger / pace, for & against)
and add interaction features (home strength x away weakness minus the reverse).
If matchups matter beyond overall quality, these improve LOSO log-loss. The spec
warns they usually collapse into quality — this tests it.

    python -m scripts.test_matchup
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import availability, games as G, goalie, predict, rating, rest, special

ROOT = Path(__file__).resolve().parent.parent
PRIOR = "20202021"
TEST = ["20212022", "20222023", "20232024"]
BASE = ["goalie_diff", "rest_diff", "st_diff", "linedev_diff"]
DIMS = ["rush", "reb", "hd", "pace"]
PC = 200   # shrink pseudo-attempts toward league rate


def load_events(seasons):
    return pd.concat([pd.read_parquet(ROOT / "data" / f"events_{s}.parquet").assign(season=str(s))
                      for s in seasons if (ROOT / "data" / f"events_{s}.parquet").exists()],
                     ignore_index=True)


def style_features(events, allg):
    """Walk-forward per-game home/away style tendencies + interaction terms."""
    ev = events.copy()
    ev["rush"] = ev["is_rush"].fillna(False).astype(int)
    ev["reb"] = ev["is_rebound"].fillna(False).astype(int)
    ev["hd"] = (ev["danger"] == "high").astype(int)
    # for-side: by attacking team; against-side: by defending team (opponent)
    forr = ev.groupby(["game_id", "team"]).agg(
        att=("is_shot_attempt", "size"), rush=("rush", "sum"),
        reb=("reb", "sum"), hd=("hd", "sum")).reset_index()
    agn = ev.groupby(["game_id", "opponent"]).agg(
        att=("is_shot_attempt", "size"), rush=("rush", "sum"),
        reb=("reb", "sum"), hd=("hd", "sum")).reset_index().rename(columns={"opponent": "team"})
    meta = allg[["game_id", "date", "season"]]
    forr = forr.merge(meta, on="game_id").sort_values(["date", "game_id"])
    agn = agn.merge(meta, on="game_id").sort_values(["date", "game_id"])

    lg = {d: ev[d].sum() / len(ev) for d in ("rush", "reb", "hd")}
    lg_pace = forr["att"].mean()

    def walk(df, side):
        cum = {}          # team -> [att, rush, reb, hd]
        rows = {}
        for r in df.itertuples(index=False):
            t = r.team
            c = cum.get(t, [0, 0, 0, 0])
            a = c[0]
            rows[(r.game_id, t)] = {
                f"{side}_rush": (c[1] + lg["rush"] * PC) / (a + PC),
                f"{side}_reb": (c[2] + lg["reb"] * PC) / (a + PC),
                f"{side}_hd": (c[3] + lg["hd"] * PC) / (a + PC),
                f"{side}_pace": (a / max(_gp(cum, t), 1)) if a else lg_pace,
            }
            cum[t] = [a + r.att, c[1] + r.rush, c[2] + r.reb, c[3] + r.hd]
            _bump(cum, t)
        return rows
    # pace needs games played; track separately via a simple counter
    return forr, agn, lg, lg_pace


# (helpers kept trivial: pace approximated by attempts/PC-scaled below instead)
def _gp(cum, t):
    return 1
def _bump(cum, t):
    pass


def main():
    seasons = ([PRIOR] if (ROOT / "data" / f"events_{PRIOR}.parquet").exists() else []) + TEST
    events = load_events(seasons)
    gbs = {s: G.load_or_build(s) for s in seasons}
    allg = pd.concat([gbs[s] for s in seasons], ignore_index=True)

    preds, _ = rating.walk_forward(events, gbs, seasons, situation="all")
    preds = preds.merge(goalie.walk_forward_goalie(events, gbs, seasons)[["game_id", "goalie_diff"]], on="game_id", how="left")
    preds = preds.merge(rest.rest_features(gbs, seasons)[["game_id", "rest_diff"]], on="game_id", how="left")
    preds = preds.merge(special.walk_forward_st(events, gbs, seasons), on="game_id", how="left")
    lu = [pd.read_parquet(ROOT / "data" / f"lineups_{s}.parquet") for s in seasons
          if (ROOT / "data" / f"lineups_{s}.parquet").exists()]
    preds = preds.merge(availability.walk_forward_availability(pd.concat(lu), gbs, seasons), on="game_id", how="left")

    # ---- walk-forward style tendencies (for & against), then interactions ----
    ev = events.copy()
    ev["rush"] = ev["is_rush"].fillna(False).astype(int)
    ev["reb"] = ev["is_rebound"].fillna(False).astype(int)
    ev["hd"] = (ev["danger"] == "high").astype(int)
    forr = ev.groupby(["game_id", "team"]).agg(att=("is_shot_attempt", "size"), rush=("rush", "sum"), reb=("reb", "sum"), hd=("hd", "sum")).reset_index()
    agn = ev.groupby(["game_id", "opponent"]).agg(att=("is_shot_attempt", "size"), rush=("rush", "sum"), reb=("reb", "sum"), hd=("hd", "sum")).reset_index().rename(columns={"opponent": "team"})
    meta = allg[["game_id", "date", "season"]].drop_duplicates()
    lg = {d: ev[d].mean() for d in ("rush", "reb", "hd")}

    def walk(df):
        df = df.merge(meta, on="game_id").sort_values(["date", "game_id"])
        cum, gp, out = {}, {}, {}
        for r in df.itertuples(index=False):
            t = r.team; c = cum.get(t, [0, 0, 0, 0]); n = gp.get(t, 0)
            out[(r.game_id, t)] = {
                "rush": (c[1] + lg["rush"] * PC) / (c[0] + PC),
                "reb": (c[2] + lg["reb"] * PC) / (c[0] + PC),
                "hd": (c[3] + lg["hd"] * PC) / (c[0] + PC),
                "pace": c[0] / n if n else float("nan"),
            }
            cum[t] = [c[0] + r.att, c[1] + r.rush, c[2] + r.reb, c[3] + r.hd]; gp[t] = n + 1
        return out
    fo, de = walk(forr), walk(agn)          # offensive tendencies, defensive-allowed
    lgpace = forr.att.mean()

    def inter(row):
        h, a = row.home, row.away
        feats = {}
        for d in DIMS:
            ho = fo.get((row.game_id, h), {}).get(d); ao = fo.get((row.game_id, a), {}).get(d)
            hd_ = de.get((row.game_id, h), {}).get(d); ad = de.get((row.game_id, a), {}).get(d)
            base = lgpace if d == "pace" else lg[d]
            if None in (ho, ao, hd_, ad) or any(pd.isna(x) for x in (ho, ao, hd_, ad)):
                feats[f"mx_{d}"] = 0.0
            else:
                feats[f"mx_{d}"] = (ho - base) * (ad - base) - (ao - base) * (hd_ - base)
        return pd.Series(feats)

    mx = allg[["game_id", "home", "away"]].join(allg.apply(inter, axis=1))
    preds = preds.merge(mx[["game_id"] + [f"mx_{d}" for d in DIMS]], on="game_id", how="left")
    MX = [f"mx_{d}" for d in DIMS]
    preds[BASE + MX] = preds[BASE + MX].fillna(0.0)

    def loso(feats):
        H = []
        for s in TEST:
            tr, te = preds[preds.season != s], preds[preds.season == s].copy()
            etr = {f: tr[f].to_numpy() for f in feats}; ete = {f: te[f].to_numpy() for f in feats}
            mm = predict.MarginMap().fit(tr.pred_xg_margin, tr.home_margin, etr)
            p = np.clip(mm.p_home_win(te.pred_xg_margin, ete), 1e-6, 1 - 1e-6); y = te.home_win.to_numpy()
            te["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            te["ok"] = ((mm.margin(te.pred_xg_margin, ete) > 0).astype(int) == y).astype(int)
            H.append(te)
        R = pd.concat(H); return R.ll.mean(), R.ok.mean()

    l0, a0 = loso(BASE)
    l1, a1 = loso(BASE + MX)
    print(f"full model              : logloss={l0:.4f} acc={a0:.3f}")
    print(f"full + style matchups   : logloss={l1:.4f} acc={a1:.3f}")
    mm = predict.MarginMap().fit(preds.pred_xg_margin, preds.home_margin, {f: preds[f].to_numpy() for f in BASE + MX})
    print("matchup coefficients:", {f: round(b, 3) for f, b in zip(mm.features, mm.betas) if f.startswith("mx_")})


if __name__ == "__main__":
    main()
