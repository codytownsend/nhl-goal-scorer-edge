"""DISCOVERY ENGINE (step 1): emergent player STYLES from raw play, goal-blind.

We never look at goals here. For each (player, season) we build an action
profile purely from HOW they shoot -- where from, what type, rush/rebound,
quality, strength state, volume, accuracy -- then LEARN a compressed embedding
(autoencoder) and let styles EMERGE as clusters. We don't define the styles.

The gate that makes it real: a style is only meaningful if it's STABLE --
a player should land in ~the same style next season. We measure that vs chance.

Outputs data/player_styles.parquet (player, season, embedding, style) +
prints the cluster fingerprints and the stability result.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025",
           "20252026"]
MIN_ATT = 60          # min shot attempts in a season for a stable profile
SHOT_TYPES = ["wrist", "slap", "snap", "tip-in", "backhand", "deflected",
              "wrap-around"]


def profile_season(e):
    """One goal-blind action-style row per shooter for a season."""
    e = e.dropna(subset=["shooter_id"]).copy()
    e["shooter_id"] = e["shooter_id"].astype(int)
    st = e["shot_type"].where(e["shot_type"].isin(SHOT_TYPES), "other")
    # precompute per-event booleans, then a single vectorized groupby-mean
    b = pd.DataFrame({
        "shooter_id": e["shooter_id"].values,
        "ongoal_sh": e["event"].isin(["shot-on-goal", "goal"]).values,
        "blocked_sh": (e["event"] == "blocked-shot").values,
        "dist_mean": e["distance"].values,
        "angle_mean": e["angle"].abs().values,
        "dngr_high": (e["danger"] == "high").values,
        "dngr_mid": (e["danger"] == "mid").values,
        "rush_sh": e["is_rush"].values,
        "reb_sh": e["is_rebound"].values,
        "xg_mean": e["xg"].values,
        "pp_sh": (e["strength_bucket"] == "PP").values,
        "pk_sh": (e["strength_bucket"] == "PK").values,
        "quick_sh": (e["secs_since_last"] < 4).values,
    })
    for s in SHOT_TYPES:
        b[f"st_{s}"] = (st == s).values
    f = b.groupby("shooter_id").mean()
    f["att"] = e.groupby("shooter_id").size()
    f["games"] = e.groupby("shooter_id")["game_id"].nunique()
    f["att_pg"] = f["att"] / f["games"]
    f = f[f["att"] >= MIN_ATT].copy()
    return f.reset_index()


def autoencoder_embed(X, dim=8, seed=0, epochs=400):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    Xt = torch.tensor(X, dtype=torch.float32)
    p = X.shape[1]
    enc = torch.nn.Sequential(torch.nn.Linear(p, 24), torch.nn.ReLU(),
                              torch.nn.Linear(24, dim))
    dec = torch.nn.Sequential(torch.nn.Linear(dim, 24), torch.nn.ReLU(),
                              torch.nn.Linear(24, p))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        z = enc(Xt)
        loss = torch.nn.functional.mse_loss(dec(z), Xt)
        loss.backward(); opt.step()
    with torch.no_grad():
        return enc(Xt).numpy()


def main():
    rows = []
    for s in SEASONS:
        try:
            e = pd.read_parquet(f"data/events_{s}.parquet")
        except FileNotFoundError:
            continue
        p = profile_season(e)
        p["season"] = int(s)
        rows.append(p)
    prof = pd.concat(rows, ignore_index=True)
    feat_cols = [c for c in prof.columns
                 if c not in ("shooter_id", "season", "att", "games")]
    print(f"profiles: {len(prof)} player-seasons, {len(feat_cols)} features")

    # standardize across the whole pool, then LEARN the embedding (goal-blind)
    X = StandardScaler().fit_transform(prof[feat_cols].fillna(0).values)
    try:
        Z = autoencoder_embed(X, dim=8)
        emb = "autoencoder(8)"
    except Exception as ex:
        Z = PCA(n_components=8, random_state=0).fit_transform(X)
        emb = f"PCA(8) [AE unavailable: {ex}]"
    print(f"embedding: {emb}")

    # pick k by silhouette
    best = None
    for k in range(4, 10):
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        print(f"  k={k} silhouette={sil:.3f}")
        if best is None or sil > best[1]:
            best = (k, sil, km)
    k, sil, km = best
    prof["style"] = km.labels_
    print(f"\nchosen k={k} (silhouette {sil:.3f})")

    # merge position (for CHARACTERIZING only, not used to cluster)
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    pos = sp.groupby("player_id")["is_def"].mean().rename("is_def_frac")
    prof = prof.merge(pos, left_on="shooter_id", right_index=True, how="left")

    # fingerprint each emergent style (z-scored deviation of its mean profile)
    Xz = pd.DataFrame(X, columns=feat_cols)
    Xz["style"] = km.labels_
    centers = Xz.groupby("style").mean()
    print("\n=== emergent style fingerprints (top distinguishing actions) ===")
    for st in range(k):
        n = (prof["style"] == st).sum()
        dfrac = prof.loc[prof["style"] == st, "is_def_frac"].mean()
        top = centers.loc[st].reindex(
            centers.loc[st].abs().sort_values(ascending=False).index).head(6)
        tags = ", ".join(f"{c}{'+' if v > 0 else '-'}{abs(v):.1f}"
                         for c, v in top.items())
        print(f"  style {st}: n={n:>4}  D-frac={dfrac:.2f}  | {tags}")

    # STABILITY GATE: do players keep their style next season?
    prof = prof.sort_values(["shooter_id", "season"])
    piv = prof.pivot_table(index="shooter_id", columns="season",
                           values="style", aggfunc="first")
    seasons = sorted(prof.season.unique())
    same, tot = 0, 0
    for a, b in zip(seasons[:-1], seasons[1:]):
        if a in piv and b in piv:
            m = piv[[a, b]].dropna()
            same += (m[a] == m[b]).sum(); tot += len(m)
    pk = prof["style"].value_counts(normalize=True)
    chance = (pk ** 2).sum()
    print(f"\n=== STABILITY GATE ===")
    print(f"consecutive-season pairs: {tot}")
    print(f"stayed in SAME style: {same/tot:.3f}  (chance {chance:.3f}, "
          f"lift {same/tot/chance:.2f}x)")

    prof[["shooter_id", "season", "style", "is_def_frac"] + feat_cols
         ].to_parquet("data/player_styles.parquet", index=False)
    print("\nwrote data/player_styles.parquet")


if __name__ == "__main__":
    main()
