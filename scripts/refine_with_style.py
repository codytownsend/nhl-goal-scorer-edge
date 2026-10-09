"""Does adding STYLE as CONTEXT to one full model (no fragmenting) improve picks?

Earlier STYLE+R trained SEPARATE per-style models -> fragmented data -> worse.
The sound version: keep ONE model on all the data and add style as extra
features (style one-hot + the goal-blind action-profile metrics) on top of the
full goal-scoring feature set. The model uses style where it helps, ignores it
otherwise. Compared to the working baseline on the identical pool.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from style_vs_working import FEATURES, RECENT, DEV, TEST, gbm, topk


def main():
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    sp["season"] = sp["season"].astype(int)
    beh = pd.read_parquet("data/recent_behavior.parquet")
    sty = pd.read_parquet("data/player_styles.parquet")

    # style CONTEXT = the goal-blind action-profile metrics + style one-hot
    style_feats = [c for c in sty.columns
                   if c not in ("shooter_id", "season", "style")]
    oh = pd.get_dummies(sty["style"].astype(int), prefix="sty")
    sty = pd.concat([sty, oh], axis=1)
    style_ctx = style_feats + list(oh.columns)

    keep_beh = ["game_id", "player_id", "season"] + \
        [c for c in RECENT if c not in sp.columns]
    pool = (sp[sp.season.isin(TEST)]
            .merge(beh[keep_beh], on=["game_id", "player_id", "season"])
            .merge(sty[["shooter_id", "season"] + style_ctx],
                   left_on=["player_id", "season"],
                   right_on=["shooter_id", "season"]))
    recent_only = [c for c in RECENT if c not in FEATURES]
    full = FEATURES + recent_only + style_ctx
    pool = pool.dropna(subset=FEATURES + recent_only).copy()
    print(f"pool: {len(pool)} player-games, {pool.game_id.nunique()} games, "
          f"base scored {pool.scored.mean():.3f}\n")

    # BASELINE: working model, FEATURES only, DEV-trained
    dev = sp[sp.season.isin(DEV)]
    base = gbm().fit(dev[FEATURES].astype(float), dev["scored"].values)
    pool["p_base"] = base.predict_proba(pool[FEATURES].astype(float))[:, 1]

    # candidate models trained on test via both-direction split
    cands = {"FEAT+RECENT": FEATURES + recent_only,
             "FEAT+STYLECTX": FEATURES + style_ctx,
             "FEAT+RECENT+STYLECTX": full}
    for col in cands:
        pool["p_" + col] = np.nan
    for tr_s, te_s in [(TEST[0], TEST[1]), (TEST[1], TEST[0])]:
        tr, mask = pool[pool.season == tr_s], pool.season == te_s
        te = pool[mask]
        for col, feats in cands.items():
            m = gbm().fit(tr[feats].astype(float), tr["scored"].values)
            pool.loc[mask, "p_" + col] = m.predict_proba(
                te[feats].astype(float))[:, 1]

    from sklearn.metrics import log_loss
    print(f"{'model':<24}{'top1':>8}{'top3':>8}{'logloss':>10}")
    for lbl, col in [("BASELINE (rate only)", "p_base"),
                     ("FEAT+RECENT", "p_FEAT+RECENT"),
                     ("FEAT+STYLECTX", "p_FEAT+STYLECTX"),
                     ("FEAT+RECENT+STYLECTX", "p_FEAT+RECENT+STYLECTX")]:
        t1, _ = topk(pool, col, 1)
        t3, _ = topk(pool, col, 3)
        ll = log_loss(pool["scored"].values, pool[col].values, labels=[0, 1])
        print(f"{lbl:<24}{t1:>8.3f}{t3:>8.3f}{ll:>10.4f}")

    # per-season replication for the promising FEAT+STYLECTX vs baseline
    print("\nper-season top1 / top3 (replication gate):")
    for sv in TEST:
        s = pool[pool.season == sv]
        b1, _ = topk(s, "p_base", 1); b3, _ = topk(s, "p_base", 3)
        c1, _ = topk(s, "p_FEAT+STYLECTX", 1)
        c3, _ = topk(s, "p_FEAT+STYLECTX", 3)
        print(f"  {str(sv)[:4]} (n={s.game_id.nunique()}): "
              f"baseline {b1:.3f}/{b3:.3f}   "
              f"FEAT+STYLECTX {c1:.3f}/{c3:.3f}   "
              f"(d top1 {c1-b1:+.3f}, top3 {c3-b3:+.3f})")


if __name__ == "__main__":
    main()
