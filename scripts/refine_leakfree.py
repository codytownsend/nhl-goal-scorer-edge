"""Leak-free version of the style-context refinement.

The earlier FEAT+STYLECTX used each player's FULL-SEASON action profile -> that
peeks at games after the one being predicted. Here we attach each player's style
profile from the PRIOR season only (strictly past), which is legitimate because
style is stable year-to-year (77.7% persistence). Players with no qualifying
prior season get no style context (GBM handles the missing values).

Compares the working baseline vs FEAT + prior-season STYLECTX, with the
both-season replication gate.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from style_vs_working import FEATURES, DEV, TEST, gbm, topk

SEASON_ORDER = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]
NEXT = {SEASON_ORDER[i]: SEASON_ORDER[i + 1]
        for i in range(len(SEASON_ORDER) - 1)}


def main():
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    sp["season"] = sp["season"].astype(int)
    sty = pd.read_parquet("data/player_styles.parquet")

    style_feats = [c for c in sty.columns
                   if c not in ("shooter_id", "season", "style")]
    oh = pd.get_dummies(sty["style"].astype(int), prefix="sty").astype(float)
    sty = pd.concat([sty, oh], axis=1)
    style_ctx = style_feats + list(oh.columns)

    # PRIOR-SEASON style: a profile from season S applies to games in season S+1
    prior = sty[["shooter_id", "season"] + style_ctx].copy()
    prior["apply_season"] = prior["season"].map(NEXT)
    prior = prior.dropna(subset=["apply_season"])
    prior["apply_season"] = prior["apply_season"].astype(int)

    pool = (sp[sp.season.isin(TEST)]
            .merge(prior.drop(columns="season"),
                   left_on=["player_id", "season"],
                   right_on=["shooter_id", "apply_season"], how="left"))
    pool = pool.dropna(subset=FEATURES).copy()
    # one-hot missing -> 0 (unknown style); numeric style feats stay NaN for GBM
    for c in oh.columns:
        pool[c] = pool[c].fillna(0.0)
    cov = pool[style_feats[0]].notna().mean()
    print(f"pool: {len(pool)} player-games, {pool.game_id.nunique()} games, "
          f"base scored {pool.scored.mean():.3f}")
    print(f"prior-season style coverage: {cov:.1%} of player-games\n")

    # BASELINE: working model, FEATURES only, DEV-trained
    dev = sp[sp.season.isin(DEV)]
    base = gbm().fit(dev[FEATURES].astype(float), dev["scored"].values)
    pool["p_base"] = base.predict_proba(pool[FEATURES].astype(float))[:, 1]

    # FEAT + prior-season STYLECTX, both-direction split on test
    feats = FEATURES + style_ctx
    pool["p_style"] = np.nan
    for tr_s, te_s in [(TEST[0], TEST[1]), (TEST[1], TEST[0])]:
        tr, mask = pool[pool.season == tr_s], pool.season == te_s
        m = gbm().fit(tr[feats].astype(float), tr["scored"].values)
        pool.loc[mask, "p_style"] = m.predict_proba(
            pool[mask][feats].astype(float))[:, 1]

    print(f"{'model':<28}{'top1':>8}{'top3':>8}{'logloss':>10}")
    for lbl, col in [("BASELINE (rate only)", "p_base"),
                     ("FEAT + prior STYLECTX", "p_style")]:
        t1, _ = topk(pool, col, 1)
        t3, _ = topk(pool, col, 3)
        ll = log_loss(pool["scored"].values, pool[col].values, labels=[0, 1])
        print(f"{lbl:<28}{t1:>8.3f}{t3:>8.3f}{ll:>10.4f}")

    print("\nper-season top1 / top3 (replication gate):")
    for sv in TEST:
        s = pool[pool.season == sv]
        b1, _ = topk(s, "p_base", 1); b3, _ = topk(s, "p_base", 3)
        c1, _ = topk(s, "p_style", 1); c3, _ = topk(s, "p_style", 3)
        print(f"  {str(sv)[:4]} (n={s.game_id.nunique()}): "
              f"baseline {b1:.3f}/{b3:.3f}   prior-STYLECTX {c1:.3f}/{c3:.3f}   "
              f"(d top1 {c1-b1:+.3f}, top3 {c3-b3:+.3f})")


if __name__ == "__main__":
    main()
