"""EV check WITHOUT historical odds: is the STYLE model's different pick a
longer shot that beats its fair price?

Logic (no odds needed): our baseline scorer model is well-calibrated and the
market tracks a calibrated model, so a player's baseline P(goal) ~= his fair
(no-vig) implied probability. So:
  Part 1 - how much longer a shot is the style pick? (compare baseline P of each
           model's divergent top-1 pick)
  Part 2 - calibration-edge: does the style pick SCORE more often than its fair
           (baseline) probability? hit_rate - fair_prob = value before vig.
           Done per season (replication) + per style.
  Part 3 - vig overlay from the one real cached slate (props_2026-10-08.json):
           typical decimal / implied prob at the longshot tier, so we know how
           much positive value is needed to clear the juice.
"""
import json
import numpy as np
import pandas as pd

from style_vs_working import build_predictions, TEST


def top1_per_game(pool, col):
    idx = pool.groupby("game_id")[col].idxmax()
    return pool.loc[idx, ["game_id", "player_id", "scored", "p_base", "season",
                          "style"]].set_index("game_id")


def main():
    pool = build_predictions()
    b = top1_per_game(pool, "p_base")
    s = top1_per_game(pool, "p_style")
    j = b.join(s, lsuffix="_b", rsuffix="_s")
    diff = j[j.player_id_b != j.player_id_s]
    print(f"games {len(j)}, disagreements {len(diff)} ({len(diff)/len(j):.1%})")

    # Part 1: how much longer a shot is the style pick?
    print("\n--- PART 1: is the STYLE pick a longer shot? (baseline fair prob) ---")
    print(f"  baseline-pick fair prob : mean {diff.p_base_b.mean():.3f}  "
          f"median {diff.p_base_b.median():.3f}")
    print(f"  style-pick    fair prob : mean {diff.p_base_s.mean():.3f}  "
          f"median {diff.p_base_s.median():.3f}")
    print(f"  => style reaches {diff.p_base_b.mean()-diff.p_base_s.mean():.3f} "
          f"lower-probability on average")

    # Part 2: calibration-edge. Does each pick beat its fair (baseline) prob?
    print("\n--- PART 2: do the picks beat their FAIR price? "
          "(hit_rate - fair_prob) ---")
    print(f"  {'set':<26}{'n':>6}{'hit':>8}{'fair':>8}{'value':>8}")
    for name, sub in [("ALL disagreements", diff)] + \
            [(f"  season {str(sv)[:4]}", diff[diff.season_b == sv])
             for sv in TEST]:
        for who, hit_c, fair_c in [("baseline pick", "scored_b", "p_base_b"),
                                   ("style pick", "scored_s", "p_base_s")]:
            hit, fair = sub[hit_c].mean(), sub[fair_c].mean()
            print(f"  {name+' / '+who:<26}{len(sub):>6}{hit:>8.3f}"
                  f"{fair:>8.3f}{hit-fair:>+8.3f}")
    # per style (style pick only)
    print("\n  style pick value by STYLE (both seasons):")
    for st in sorted(diff.style_s.dropna().unique()):
        d = diff[diff.style_s == st]
        if len(d) < 50:
            continue
        print(f"    style {int(st)}: n={len(d):>4}  hit {d.scored_s.mean():.3f}"
              f"  fair {d.p_base_s.mean():.3f}  value {d.scored_s.mean()-d.p_base_s.mean():+.3f}")

    # Part 3: vig overlay from the real cached slate
    print("\n--- PART 3: vig overlay (real slate props_2026-10-08.json) ---")
    try:
        raw = json.load(open("data/raw/props_2026-10-08.json"))
        dec = []  # decimal odds (prices are AMERICAN)
        for game in raw.get("events", []):
            for bk in game.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if "goal_scorer_anytime" in mk.get("key", ""):
                        for o in mk.get("outcomes", []):
                            a = o.get("price")
                            if a and o.get("name") == "Yes":
                                dec.append(1 + a / 100 if a > 0 else 1 + 100 / -a)
        dec = np.array(dec)
        implied = 1 / dec  # includes vig
        ls = dec >= 3.5    # longshot tier ~ +250 or longer
        print(f"  {len(dec)} 'Yes' scorer prices; mean decimal {dec.mean():.2f}, "
              f"mean implied(w/ vig) {implied.mean():.3f}")
        if ls.any():
            print(f"  longshot tier (decimal>=3.5): n={ls.sum()}, mean implied "
                  f"{implied[ls].mean():.3f}, mean decimal {dec[ls].mean():.2f}")
            print(f"  => at that tier realized hit-rate must exceed ~"
                  f"{implied[ls].mean():.3f} to beat the vig")
    except Exception as ex:
        print(f"  (could not parse slate: {ex})")


if __name__ == "__main__":
    main()
