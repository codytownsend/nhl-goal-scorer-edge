"""Ensemble bake-off: earlier xG model (A) vs new xEV model (B), and every
combination — blends, optimized weight, logistic stack, one combined model, and
confidence/agreement gating. All scored out-of-sample (2024-26).

    python -m scripts.ensemble
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import event_value as EV, games as G, predict
from nhlsit.xg import _irls_logistic, _sigmoid
from scripts.backtest_oos import build_preds

ROOT = Path(__file__).resolve().parent.parent
DEV = ["20202021", "20212022", "20222023", "20232024"]
OOS = ["20242025", "20252026"]
ALL = DEV + OOS
FEATS = ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "finish_diff"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    preds = build_preds()
    gbs = {s: G.load_or_build(s) for s in ALL}
    ae = pd.concat([pd.read_parquet(ROOT / f"data/allevents_{s}.parquet") for s in ALL], ignore_index=True)
    ae["season"] = ae["season"].astype(str)
    values = EV.learn_values(ae[ae.season.isin(DEV)])
    xev = EV.walk_forward_xev(ae, gbs, ALL, values)
    m = preds.merge(xev[["game_id", "pred_xev_margin"]], on="game_id")

    dev, oos = m[m.season.isin(DEV)], m[m.season.isin(OOS)]
    yd, yo = dev.home_win.to_numpy(), oos.home_win.to_numpy()

    # models A (xG+feats) and B (xEV+feats)
    A = predict.MarginMap().fit(dev.pred_xg_margin, dev.home_margin, {f: dev[f].to_numpy() for f in FEATS})
    B = predict.MarginMap().fit(dev.pred_xev_margin, dev.home_margin, {f: dev[f].to_numpy() for f in FEATS})
    exd = {f: dev[f].to_numpy() for f in FEATS}; exo = {f: oos[f].to_numpy() for f in FEATS}
    pA_d, pA_o = A.p_home_win(dev.pred_xg_margin, exd), A.p_home_win(oos.pred_xg_margin, exo)
    pB_d, pB_o = B.p_home_win(dev.pred_xev_margin, exd), B.p_home_win(oos.pred_xev_margin, exo)

    def ev(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        acc = ((p >= 0.5).astype(int) == yo).mean()
        ll = -np.mean(yo * np.log(p) + (1 - yo) * np.log(1 - p))
        return acc, ll

    print(f"{'combination':34} {'acc':>7} {'logloss':>9}")
    print(f"{'A: xG model (earlier)':34} {ev(pA_o)[0]:7.3f} {ev(pA_o)[1]:9.4f}")
    print(f"{'B: xEV model (new)':34} {ev(pB_o)[0]:7.3f} {ev(pB_o)[1]:9.4f}")
    print(f"{'mean(pA,pB)':34} {ev((pA_o+pB_o)/2)[0]:7.3f} {ev((pA_o+pB_o)/2)[1]:9.4f}")
    print(f"{'logit-average':34} {ev(_sigmoid((logit(pA_o)+logit(pB_o))/2))[0]:7.3f} {ev(_sigmoid((logit(pA_o)+logit(pB_o))/2))[1]:9.4f}")

    # optimal weight on DEV (min logloss)
    ws = np.linspace(0, 1, 21); best = None
    for w in ws:
        pd_ = np.clip(w * pA_d + (1 - w) * pB_d, 1e-6, 1 - 1e-6)
        ll = -np.mean(yd * np.log(pd_) + (1 - yd) * np.log(1 - pd_))
        if best is None or ll < best[0]:
            best = (ll, w)
    w = best[1]; pblend = w * pA_o + (1 - w) * pB_o
    print(f"{'weighted blend (w*A, w=%.2f)'%w:34} {ev(pblend)[0]:7.3f} {ev(pblend)[1]:9.4f}")

    # logistic stack on [logit pA, logit pB], fit DEV
    Xd = np.column_stack([np.ones(len(dev)), logit(pA_d), logit(pB_d)])
    Xo = np.column_stack([np.ones(len(oos)), logit(pA_o), logit(pB_o)])
    wts = _irls_logistic(Xd, yd.astype(float))
    print(f"{'logistic stack (A,B)':34} {ev(_sigmoid(Xo@wts))[0]:7.3f} {ev(_sigmoid(Xo@wts))[1]:9.4f}  coefs={np.round(wts,2)}")

    # one combined MarginMap: xG primary + xEV as linear feature + feats
    C = predict.MarginMap().fit(dev.pred_xg_margin, dev.home_margin,
                                {**{f: dev[f].to_numpy() for f in FEATS}, "xev": dev.pred_xev_margin.to_numpy()})
    pC = C.p_home_win(oos.pred_xg_margin, {**exo, "xev": oos.pred_xev_margin.to_numpy()})
    print(f"{'combined model (xG+xEV+feats)':34} {ev(pC)[0]:7.3f} {ev(pC)[1]:9.4f}")

    # confidence: use the more-confident model's pick
    confA, confB = np.abs(pA_o - .5), np.abs(pB_o - .5)
    p_moreconf = np.where(confA >= confB, pA_o, pB_o)
    print(f"{'pick more-confident model':34} {ev(p_moreconf)[0]:7.3f} {ev(p_moreconf)[1]:9.4f}")

    # agreement gating
    agree = (pA_o >= .5) == (pB_o >= .5)
    accA_ag = (((pA_o[agree] >= .5).astype(int)) == yo[agree]).mean()
    accA_dis = (((pA_o[~agree] >= .5).astype(int)) == yo[~agree]).mean()
    print(f"\nagreement: A&B agree on {agree.mean():.0%} of games")
    print(f"  acc when they AGREE:    {accA_ag:.3f} (n={agree.sum()})")
    print(f"  acc when they DISAGREE: {accA_dis:.3f} (n={(~agree).sum()})")
    print(f"\ncorr(pA,pB) = {np.corrcoef(pA_o,pB_o)[0,1]:.3f}")


if __name__ == "__main__":
    main()
