"""Compare model predictions to the moneyline market.

Requires an odds CSV (columns: date, home, away, home_ml, away_ml) and the LOSO
predictions written by scripts.evaluate.

    python -m scripts.evaluate                      # writes data/loso_predictions.parquet
    python -m scripts.market_eval path/to/odds.csv

Odds sources (the model never uses them as input; benchmark only):
  - historical: any CSV you have / build with the columns above
  - live upcoming: nhlsit.market.fetch_live_odds() via The Odds API free tier
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nhlsit import market

ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    odds_path = sys.argv[1]
    preds_fp = ROOT / "data" / "loso_predictions.parquet"
    if not preds_fp.exists():
        raise SystemExit("run `python -m scripts.evaluate` first")

    preds = pd.read_parquet(preds_fp)
    cols = {c.lower() for c in pd.read_csv(odds_path, nrows=1).columns}
    if {"a__team", "h__team", "fav"} <= cols:          # favorite-only feed
        odds = market.load_favorite_odds_csv(odds_path)
        print("(favorite-only odds feed: underdog price reconstructed at 4.5% hold)")
    else:
        odds = market.load_odds_csv(odds_path)
    res = market.evaluate_vs_market(preds, odds)

    print(f"\nmatched games: {res['n_games']:,}\n")
    print(f"  model straight-up acc : {res['model_acc']:.3f}")
    print(f"  market straight-up acc: {res['market_acc']:.3f}")
    print(f"  disagreements         : {res['n_disagree']}  "
          f"(model right on them: {res['model_acc_on_disagree']:.3f})")
    print("\n  moneyline +EV bets (flat 1u):")
    print(f"    bets      : {res['n_bets']}")
    print(f"    hit rate  : {res['bet_hit_rate']:.3f}")
    print(f"    ROI       : {res['roi']:+.3%}")
    print(f"    net units : {res['units']:+.1f}")

    out = ROOT / "data" / "market_bet_ledger.csv"
    res["ledger"].to_csv(out, index=False)
    print(f"\nwrote bet ledger -> {out}\n")


if __name__ == "__main__":
    main()
