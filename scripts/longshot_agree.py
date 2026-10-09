"""Does main-model ENDORSEMENT rescue the style model's longshot picks?

The style model's top-1 pick is often a longshot (lower fair prob). On its own
those picks only hit at their fair price (no edge). This asks: when the main
(baseline) model ALSO likes that same longshot -- as its own top-1, or within
its top-3 -- does the longshot become reliably correct / beat its fair price?

"Longshot" = the style pick's baseline P(goal) (our fair, no-vig price proxy)
below a threshold. We report hit-rate and value (hit - fair) for:
  - style pick is a longshot AND == baseline top-1
  - style pick is a longshot AND in baseline top-3
  - style pick is a longshot AND NOT in baseline top-3  (the contrast)
per season (replication) and pooled.
"""
import numpy as np
import pandas as pd

from style_vs_working import build_predictions, TEST


def per_game_picks(pool):
    rows = []
    for gid, g in pool.groupby("game_id"):
        g = g.sort_values("p_base", ascending=False)
        base_top1 = g["player_id"].iloc[0]
        base_top3 = set(g["player_id"].iloc[:3])
        g = g.reset_index(drop=True)
        g["base_rank"] = g["p_base"].rank(ascending=False, method="first")
        srow = g.loc[g["p_style"].idxmax()]
        rows.append({
            "game_id": gid, "season": srow["season"],
            "pick": srow["player_id"], "scored": srow["scored"],
            "fair": srow["p_base"],               # main-model prob = fair price
            "base_rank": int(srow["base_rank"]),  # style pick's rank in main model
            "n_players": len(g),
            "is_base_top1": srow["player_id"] == base_top1,
            "is_base_top3": srow["player_id"] in base_top3,
        })
    return pd.DataFrame(rows)


def line(name, d):
    if len(d) == 0:
        print(f"  {name:<42}{'n=0':>6}")
        return
    hit, fair = d["scored"].mean(), d["fair"].mean()
    print(f"  {name:<42}{len(d):>6}{hit:>8.3f}{fair:>8.3f}{hit-fair:>+8.3f}")


def main():
    pool = build_predictions()
    pk = per_game_picks(pool)
    print(f"games: {len(pk)}  (style top-1 pick per game)")
    print(f"style top-1 overall hit: {pk.scored.mean():.3f}\n")

    # where do the style LONGSHOT picks rank in the MAIN model?
    print("=== main-model RANK of the style pick, by longshot threshold ===")
    print(f"  {'threshold':<16}{'n':>6}{'hit':>7}{'in_top3':>9}{'in_top5':>9}"
          f"{'med_rank':>9}")
    for thr in [1.0, 0.25, 0.20, 0.15]:
        ls = pk[pk["fair"] < thr]
        tag = "ALL picks" if thr == 1.0 else f"fair<{thr:.2f}"
        print(f"  {tag:<16}{len(ls):>6}{ls.scored.mean():>7.3f}"
              f"{(ls.base_rank <= 3).mean():>9.3f}{(ls.base_rank <= 5).mean():>9.3f}"
              f"{ls.base_rank.median():>9.0f}")
    print()

    for thr in [0.15, 0.20, 0.25]:
        ls = pk[pk["fair"] < thr]
        print(f"===== LONGSHOT = style pick fair prob < {thr:.2f}  "
              f"(n={len(ls)}, {len(ls)/len(pk):.0%} of games) =====")
        print(f"  {'bucket':<42}{'n':>6}{'hit':>8}{'fair':>8}{'value':>8}")
        line("longshot & == main top-1", ls[ls.is_base_top1])
        line("longshot & in main top-3", ls[ls.is_base_top3])
        line("longshot & NOT in main top-3 (all)", ls[~ls.is_base_top3])
        # replication across seasons for the only populated bucket
        for sv in TEST:
            line(f"  NOT in top-3 / season {str(sv)[:4]}",
                 ls[~ls.is_base_top3 & (ls.season == sv)])
        print()


if __name__ == "__main__":
    main()
