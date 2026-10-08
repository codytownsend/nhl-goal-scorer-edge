"""Situational team comparison CLI.

Examples:
    # headline comparison at 5v5
    python -m scripts.compare_teams 20232024 --a TOR --b BOS --strength 5v5

    # a team's full profile split by score state (on the PP)
    python -m scripts.compare_teams 20232024 --a EDM --by score_state --strength PP

    # league leaderboard for a metric within a situation
    python -m scripts.compare_teams 20232024 --table xGF% --strength 5v5

Situation flags (all optional, combine freely):
    --strength {5v5,PP,PK,4v4/3v3,empty_net}
    --score    {trail_2+,trail_1,tied,lead_1,lead_2+}
    --period   {P1,P2,P3,OT}
    --danger   {high,mid,low}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nhlsit import compare as C

ROOT = Path(__file__).resolve().parent.parent


def _situation(args) -> dict:
    s = {}
    if args.strength:
        s["strength_bucket"] = args.strength
    if args.score:
        s["score_state"] = args.score
    if args.period:
        s["period_label"] = args.period
    if args.danger:
        s["danger"] = args.danger
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("season")
    ap.add_argument("--a", help="team A (3-letter)")
    ap.add_argument("--b", help="team B (3-letter)")
    ap.add_argument("--by", help="split a single team's profile by this dimension "
                                 "(strength_bucket, score_state, period_label, danger)")
    ap.add_argument("--table", metavar="METRIC",
                    help="league leaderboard on METRIC (e.g. xGF%%, CF%%, HDCF%%)")
    ap.add_argument("--strength")
    ap.add_argument("--score")
    ap.add_argument("--period")
    ap.add_argument("--danger")
    args = ap.parse_args()

    fp = ROOT / "data" / f"events_{args.season}.parquet"
    if not fp.exists():
        raise SystemExit(f"No data at {fp}. Run: python -m scripts.build_events {args.season}")
    ev = pd.read_parquet(fp)
    sit = _situation(args)
    sit_txt = ", ".join(f"{k}={v}" for k, v in sit.items()) or "all situations"

    pd.set_option("display.width", 160)

    if args.table:
        print(f"\nLeague leaderboard — {args.table}  [{sit_txt}]\n")
        print(C.league_table(ev, args.table, sit).to_string())
    elif args.a and args.by:
        print(f"\n{args.a} profile by {args.by}  [{sit_txt}]\n")
        print(C.team_profile(ev, args.a, by=args.by, situation=sit).to_string())
        print(f"\n{args.a} offensive danger mix: "
              + C.danger_mix(ev, args.a, "offense", sit).to_dict().__str__())
    elif args.a and args.b:
        print(f"\n{args.a} vs {args.b}  [{sit_txt}]\n")
        print(C.compare(ev, args.a, args.b, sit).to_string())
    else:
        raise SystemExit("Give --table METRIC, or --a and --b, or --a with --by.")
    print()


if __name__ == "__main__":
    main()
