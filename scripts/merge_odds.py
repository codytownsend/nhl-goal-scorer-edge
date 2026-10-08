"""Merge the once-a-day anytime-scorer odds pull into predictions_<date>.json.

Pulls props ONCE (cached by market.fetch_scorer_props), matches each book line to
our players by last name + first initial within the same game, then fills:
  vegas_odds (median American), vegas_prob (implied, incl. vig),
  edge = our P(goal) - vegas_prob, ev = our_p * decimal_odds - 1.
Also stamps the JSON with odds_source + requests_remaining (for the dashboard).

Usage: THE_ODDS_API_KEY=... python -m scripts.merge_odds 2026-10-08
"""
import json
import sys
import unicodedata

from nhlsit import market


def norm(name: str) -> str:
    """'William Nylander' / 'W. Nylander' -> 'nylander|w' match key."""
    n = "".join(c for c in unicodedata.normalize("NFKD", str(name))
                if not unicodedata.combining(c))
    parts = n.replace(".", "").split()
    if not parts:
        return ""
    last = parts[-1].lower()
    first_i = parts[0][0].lower() if parts[0] else ""
    return f"{last}|{first_i}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else "2026-10-08"
    # --game TOR@VGK  (repeatable) -> fetch only those; else fill ALL missing games
    only = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--game"] or None
    path = f"predictions_{date}.json"
    doc = json.load(open(path))

    df, meta = market.fetch_scorer_props(date, only_games=only)
    lut = {(r.game, norm(r.player)): r.odds for r in df.itertuples(index=False)}

    matched = total = 0
    for g in doc["games"]:
        for p in g["players"]:
            total += 1
            odds = lut.get((g["game"], norm(p["name"])))
            if odds is None:
                continue
            matched += 1
            dec = market.american_to_decimal(odds)
            p["vegas_odds"] = int(odds)
            p["vegas_prob"] = round(market.american_to_prob(odds), 4)
            p["edge"] = round(p["p_goal"] - p["vegas_prob"], 4)
            p["ev"] = round(p["p_goal"] * dec - 1.0, 4)

    doc["odds_source"] = "the-odds-api: player_goal_scorer_anytime (median, incl. vig)"
    doc["requests_remaining"] = meta.get("requests_remaining")
    doc["requests_used"] = meta.get("requests_used")
    doc["odds_pulled_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    json.dump(doc, open(path, "w"), indent=2)
    still_missing = [g["game"] for g in doc["games"]
                     if all(p["vegas_odds"] is None for p in g["players"])]
    print(f"fetched {meta.get('fetched', 0)} new game(s) this call; "
          f"matched {matched}/{total} players; "
          f"requests remaining: {meta.get('requests_remaining')}")
    if still_missing:
        print(f"still no lines posted for: {', '.join(still_missing)} "
              f"(re-run later, or: --game {still_missing[0]})")
    print(f"updated {path}")


if __name__ == "__main__":
    main()
