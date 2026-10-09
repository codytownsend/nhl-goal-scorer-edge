"""Hierarchical / dynamic rate estimate (`rate_hier`) vs the current K=30
shrink-to-last-season (`rate_shrunk`).

rate_hier per player-game (all walk-forward / leak-free):
  empirical = recency-weighted (EWMA, halflife H games) career scored-rate
              using ALL prior games across seasons, pre-game.
  n         = career games before this game; effective sample N_dec from the
              geometric weights.
  prior m   = population anytime-rate for the player's (position, age-bucket,
              prior-season style) cell, learned on DEV only. Rookies anchor to
              their GROUP, not the league mean.
  rate_hier = lambda*empirical + (1-lambda)*m,  lambda = N_dec/(N_dec+kappa).

Stage 1 (this script): prove rate_hier is a better rate estimator -- higher
correlation with each player's realized same-season rate, and better standalone
top-1/top-3 -- overall and especially in the low-sample / rookie slice.
"""
import numpy as np
import pandas as pd

from style_vs_working import topk, FEATURES, gbm
from sklearn.metrics import log_loss

DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]
SEASON_ORDER = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]
NEXT = {SEASON_ORDER[i]: SEASON_ORDER[i + 1] for i in range(len(SEASON_ORDER) - 1)}
HALFLIFE = 120.0   # games; ~1.5 seasons -> gentle recency, anchored
KAPPA = 30.0       # prior pseudo-games (match current K for a fair comparison)
AGE_BINS = [0, 21, 24, 27, 30, 33, 100]


def main():
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    sp["season"] = sp["season"].astype(int)
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp.sort_values(["player_id", "date"]).reset_index(drop=True)

    # age
    bd = pd.read_parquet("data/player_birthdates.parquet")
    bd["birthDate"] = pd.to_datetime(bd["birthDate"])
    sp = sp.merge(bd, on="player_id", how="left")
    sp["age"] = (sp["date"] - sp["birthDate"]).dt.days / 365.25
    sp["age_b"] = pd.cut(sp["age"], AGE_BINS, labels=False)

    # prior-season style (leak-free)
    sty = pd.read_parquet("data/player_styles.parquet")[
        ["shooter_id", "season", "style"]].copy()
    sty["apply_season"] = sty["season"].map(NEXT)
    sty = sty.dropna(subset=["apply_season"])
    sty["apply_season"] = sty["apply_season"].astype(int)
    sp = sp.merge(sty[["shooter_id", "apply_season", "style"]],
                  left_on=["player_id", "season"],
                  right_on=["shooter_id", "apply_season"], how="left")

    # ---- recency-weighted career empirical rate (pre-game) ----
    g = sp.groupby("player_id")
    ewm = g["scored"].transform(
        lambda s: s.ewm(halflife=HALFLIFE, adjust=True).mean().shift(1))
    sp["emp"] = ewm
    sp["n_prior"] = g.cumcount()
    w = 0.5 ** (1.0 / HALFLIFE)
    sp["N_dec"] = (1 - w ** sp["n_prior"]) / (1 - w)

    # ---- group prior m from DEV: (is_def, age_b, style) w/ fallbacks ----
    dev = sp[sp.season.isin(DEV)]
    glob = dev["scored"].mean()

    def grp_mean(keys):
        return dev.groupby(keys)["scored"].mean()
    m_full = grp_mean(["is_def", "age_b", "style"])
    m_pa = grp_mean(["is_def", "age_b"])
    m_p = grp_mean(["is_def"])

    def prior_row(r):
        for tbl, key in [(m_full, (r.is_def, r.age_b, r.style)),
                         (m_pa, (r.is_def, r.age_b)), (m_p, (r.is_def,))]:
            v = tbl.get(key if len(key) > 1 else key[0])
            if v is not None and not pd.isna(v):
                return v
        return glob
    # vectorized-ish prior lookup
    idx_full = list(zip(sp.is_def, sp.age_b, sp["style"]))
    idx_pa = list(zip(sp.is_def, sp.age_b))
    pri = np.full(len(sp), glob)
    mf = m_full.to_dict(); mpa = m_pa.to_dict(); mp = m_p.to_dict()
    for i, (kf, kpa, kp) in enumerate(zip(idx_full, idx_pa, sp.is_def)):
        v = mf.get(kf)
        if v is None or pd.isna(v):
            v = mpa.get(kpa)
        if v is None or pd.isna(v):
            v = mp.get(kp, glob)
        pri[i] = v
    sp["m"] = pri

    # ---- hierarchical posterior ----
    lam = sp["N_dec"] / (sp["N_dec"] + KAPPA)
    sp["rate_hier"] = lam * sp["emp"].fillna(sp["m"]) + (1 - lam) * sp["m"]

    # realized same-season rate (the thing we're estimating) for corr benchmark
    real = (sp.groupby(["player_id", "season"])["scored"].transform("mean"))
    sp["realized"] = real

    test = sp[sp.season.isin(TEST)].copy()
    print(f"test player-games: {len(test)}  "
          f"(style coverage {test["style"].notna().mean():.0%})\n")

    print("=== correlation with realized same-season rate (higher=better) ===")
    for lbl, grp in [("ALL", test),
                     ("rookies/low-sample (<40 career g)", test[test.n_prior < 40]),
                     ("established (>=200 career g)", test[test.n_prior >= 200])]:
        c_old = grp["rate_shrunk"].corr(grp["realized"])
        c_new = grp["rate_hier"].corr(grp["realized"])
        print(f"  {lbl:<36} rate_shrunk {c_old:.3f}   rate_hier {c_new:.3f}   "
              f"({c_new-c_old:+.3f})")

    print("\n=== standalone ranking (rank games by the estimate) ===")
    print(f"  {'season':<10}{'metric':<6}{'rate_shrunk':>13}{'rate_hier':>12}")
    for sv in TEST:
        s = test[test.season == sv]
        for k in (1, 3):
            o, _ = topk(s, "rate_shrunk", k)
            n, _ = topk(s, "rate_hier", k)
            print(f"  {str(sv)[:4]:<10}top{k:<3}{o:>13.3f}{n:>12.3f}")

    # ===== STAGE 2: fold rate_hier into the FULL model (DEV-train, leak-free) =====
    print("\n=== STAGE 2: full GBM, DEV-trained, predict test ===")
    dev_rows = sp[sp.season.isin(DEV)]
    variants = {"BASELINE (FEATURES)": FEATURES,
                "FEATURES + rate_hier": FEATURES + ["rate_hier"],
                "swap rate_shrunk->rate_hier":
                    [f for f in FEATURES if f != "rate_shrunk"] + ["rate_hier"]}
    preds = {}
    for lbl, feats in variants.items():
        m = gbm().fit(dev_rows[feats].astype(float), dev_rows["scored"].values)
        test[lbl] = m.predict_proba(test[feats].astype(float))[:, 1]
        preds[lbl] = test[lbl]
    print(f"  {'model':<30}{'top1':>8}{'top3':>8}{'logloss':>10}")
    for lbl in variants:
        t1, _ = topk(test, lbl, 1); t3, _ = topk(test, lbl, 3)
        ll = log_loss(test["scored"].values, test[lbl].values, labels=[0, 1])
        print(f"  {lbl:<30}{t1:>8.3f}{t3:>8.3f}{ll:>10.4f}")
    print("  per-season top1 / top3 (replication gate):")
    for sv in TEST:
        s = test[test.season == sv]
        row = f"  {str(sv)[:4]}: "
        for lbl in variants:
            t1, _ = topk(s, lbl, 1); t3, _ = topk(s, lbl, 3)
            row += f"[{lbl.split()[0]} {t1:.3f}/{t3:.3f}] "
        print(row)


if __name__ == "__main__":
    main()
