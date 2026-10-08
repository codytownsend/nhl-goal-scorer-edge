"""Hunt for non-average structure: WHERE is the rate model repeatably wrong?

Instead of measuring a feature's GLOBAL average effect (which washes out any
pattern confined to a small slice of games), we:
  1. train the real rate model on DEV (2020-24) and predict both test seasons,
  2. compute the residual = scored - rate_prediction (what the average missed),
  3. grow a shallow tree on the residual to DISCOVER pockets -- conjunctions of
     conditions where players repeatably over- or under-score their rate,
  4. keep ONLY pockets that REPLICATE in the other season with the same sign,
     real magnitude (>=2pp), and significance (|t|>=2).

Falsifiable: surviving pockets = real patterns beyond the average; none surviving
= definitive proof the ceiling is irreducible.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

RATE_FEATS = ["g_pg", "sh_pg", "satt_pg", "ixg_pg", "hd_pg", "pp_ixg_pg", "finish",
              "toi_pg", "gp", "g_l5", "ixg_l5", "sh_l5", "g_l10", "ixg_l10",
              "is_home", "is_def", "opp_goalie", "opp_def", "rate_shrunk",
              "g_rate_shrunk", "toi_l5", "toi_trend", "pp_ixg_l5", "pp_trend"]
DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]

# conditions allowed to DEFINE a pocket (incl. the non-average, deviation signals)
COND = ["gp", "is_def", "is_home", "ixg_pg", "rate_shrunk", "opp_goalie", "opp_def",
        "finish", "toi_dev", "toi_l5", "dry5", "since_goal", "games_w_shot5",
        "ixg_dev", "hd_dev", "satt_dev", "pp_ixg_dev", "sog_dev", "onxgf_dev",
        "assist_dev", "xgpsh_dev", "sogpct_dev", "rush_dev", "reb_dev", "md_dev",
        "dist_l5", "ixg_std5"]


def leaf_rules(tree, names):
    t = tree.tree_
    out = {}
    def rec(node, conds):
        if t.children_left[node] == t.children_right[node]:
            out[node] = conds; return
        f, thr = names[t.feature[node]], t.threshold[node]
        rec(t.children_left[node], conds + [(f, "<=", thr)])
        rec(t.children_right[node], conds + [(f, ">", thr)])
    rec(0, [])
    return out


def compress(conds):
    lo, hi = {}, {}
    for f, op, thr in conds:
        if op == ">":
            lo[f] = max(lo.get(f, -1e18), thr)
        else:
            hi[f] = min(hi.get(f, 1e18), thr)
    parts = []
    for f in sorted(set(lo) | set(hi)):
        if f in lo and f in hi:
            parts.append(f"{lo[f]:.2f}<{f}<={hi[f]:.2f}")
        elif f in lo:
            parts.append(f"{f}>{lo[f]:.2f}")
        else:
            parts.append(f"{f}<={hi[f]:.2f}")
    return "  &  ".join(parts)


def hunt(fit, val, fit_nm, val_nm):
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=1000,
                                 random_state=0).fit(fit[COND], fit["resid"])
    rules = leaf_rules(tree, COND)
    fl, vl = tree.apply(fit[COND].values), tree.apply(val[COND].values)
    rows = []
    for leaf in rules:
        fm = fit.resid[fl == leaf]
        vm = val.resid[vl == leaf]
        if len(fm) < 800 or len(vm) < 300:
            continue
        se = vm.std() / np.sqrt(len(vm))
        t = vm.mean() / se if se > 0 else 0.0
        repl = (np.sign(fm.mean()) == np.sign(vm.mean())
                and abs(vm.mean()) >= 0.02 and abs(t) >= 2)
        rows.append((fm.mean(), len(fm), vm.mean(), len(vm), t, repl,
                     compress(rules[leaf])))
    rows.sort(key=lambda r: -abs(r[0]))
    print(f"\n===== fit {fit_nm}, validate {val_nm} =====")
    print(f"  {'fit_resid':>10}{'n_fit':>7}{'val_resid':>11}{'n_val':>7}"
          f"{'t_val':>7}  repl  rule")
    nrep = 0
    for fm, nf, vm, nv, t, repl, rule in rows:
        nrep += int(repl)
        mark = "**" if repl else "  "
        print(f"  {fm:>+10.3f}{nf:>7}{vm:>+11.3f}{nv:>7}{t:>+7.1f}  {mark}  {rule}")
    print(f"  -> {nrep}/{len(rows)} pockets replicate (same sign, >=2pp, |t|>=2)")
    return nrep


def main():
    df = pd.read_parquet("data/scorer_playergames.parquet")
    df["season"] = df["season"].astype(int)
    dev, test = df[df.season.isin(DEV)], df[df.season.isin(TEST)].copy()
    rate = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0).fit(dev[RATE_FEATS].astype(float),
                                                  dev["scored"])
    test["p0"] = rate.predict_proba(test[RATE_FEATS].astype(float))[:, 1]
    test["resid"] = test["scored"] - test["p0"]

    beh = pd.read_parquet("data/recent_behavior.parquet").drop(columns=["scored"])
    d = test.merge(beh, on=["game_id", "player_id", "season"], how="inner",
                   suffixes=("", "_b"))
    print(f"rows with conditions (gp>=10): {len(d)}; "
          f"global mean residual {d.resid.mean():+.4f} (≈0 = rate calibrated)")
    s1, s2 = d[d.season == TEST[0]], d[d.season == TEST[1]]
    r1 = hunt(s1, s2, "2024-25", "2025-26")
    r2 = hunt(s2, s1, "2025-26", "2024-25")
    print(f"\n==== VERDICT: {r1 + r2} replicating pockets found across both directions ====")


if __name__ == "__main__":
    main()
