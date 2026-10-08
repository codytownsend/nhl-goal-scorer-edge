"""Train the goal + point GRUs ONCE and save their weights (seed=0, reproducible).

The daily board then loads these fixed weights instead of retraining each run, so
the numbers are stable and identical on every machine. Re-run this only when you
want to refresh the model (e.g. after updating the training seasons).

Usage: python -m scripts.train_gru
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import scripts.form_poisson as FP

KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]       # drop on-ice feats -> 12
TRAIN_SEASONS = [20242025, 20252026]                # seasons in player_sequences.npz


def main():
    sq = np.load("data/player_sequences.npz", allow_pickle=True)
    Xtr = sq["X"][:, :, KEEP]
    k = pd.DataFrame({"game_id": sq["game_id"], "player_id": sq["player_id"],
                      "scored": sq["y"]})
    pts = pd.concat([pd.read_parquet(f"data/lineups_{s}.parquet")[
        ["game_id", "player_id", "points"]] for s in TRAIN_SEASONS], ignore_index=True)
    k = k.merge(pts, on=["game_id", "player_id"], how="left")
    labels = {"data/gru_goal.pt": k["scored"].values.astype(float),
              "data/gru_point.pt": (k["points"].fillna(0) >= 1).astype(float).values}
    for path, y in labels.items():
        net, mu, sd = FP.fit_gru(Xtr, y, "binary", seed=0)
        FP.save_gru(net, mu, sd, path)
        print(f"saved {path}  (trained on {len(Xtr)} sequences)")


if __name__ == "__main__":
    main()
