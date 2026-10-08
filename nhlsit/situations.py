"""Situational classification: strength state, score state, ice location.

These are the "where / when" dimensions the whole tool compares teams on.
"""
from __future__ import annotations

import math

# Net is at x = +/-89, y = 0 on the NHL coordinate system.
GOAL_X = 89.0


def parse_situation(code: str, owner_is_home: bool) -> str:
    """Map a 4-char situationCode to a strength state from the shooting
    team's perspective.

    situationCode = [away_goalie][away_skaters][home_skaters][home_goalie]
    (goalie digit: 1 = in net, 0 = pulled).
    """
    if not code or len(code) != 4 or not code.isdigit():
        return "unknown"
    ag, ask, hsk, hg = (int(c) for c in code)
    my_sk, opp_sk = (hsk, ask) if owner_is_home else (ask, hsk)
    my_g, opp_g = (hg, ag) if owner_is_home else (ag, hg)

    if my_g == 0:
        return "own_net_empty"      # shooting team pulled its goalie (down late)
    if opp_g == 0:
        return "opp_net_empty"      # empty-net chance
    if my_sk == opp_sk:
        return f"{my_sk}v{my_sk}"   # 5v5, 4v4, 3v3
    return "PP" if my_sk > opp_sk else "PK"


def strength_bucket(state: str) -> str:
    """Coarser grouping for headline comparisons."""
    if state in ("5v5",):
        return "5v5"
    if state == "PP":
        return "PP"
    if state == "PK":
        return "PK"
    if state in ("4v4", "3v3"):
        return "4v4/3v3"
    if state in ("own_net_empty", "opp_net_empty"):
        return "empty_net"
    return "other"


def score_state(diff: int) -> str:
    """Shooting team's goal differential at time of event -> bucket."""
    if diff <= -2:
        return "trail_2+"
    if diff == -1:
        return "trail_1"
    if diff == 0:
        return "tied"
    if diff == 1:
        return "lead_1"
    return "lead_2+"


def period_label(number: int, period_type: str) -> str:
    if period_type and period_type.upper() == "SO":
        return "SO"
    if period_type and period_type.upper() == "OT":
        return "OT"
    return f"P{number}"


def time_to_seconds(time_in_period: str, period_number: int) -> int:
    """Absolute game seconds elapsed (regulation periods are 1200s each)."""
    try:
        m, s = time_in_period.split(":")
        within = int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        within = 0
    return (period_number - 1) * 1200 + within


def orient(x: float, y: float, owner_is_home: bool, home_defending_side: str):
    """Flip coordinates so the shooting team always attacks the +x net.

    home_defending_side is the side (left/right) the home team defends this
    period; the attacking side is the opposite.
    """
    if x is None or y is None:
        return None, None
    home_attacks_right = (home_defending_side == "left")
    attacks_right = home_attacks_right if owner_is_home else (not home_attacks_right)
    if not attacks_right:
        return -x, -y
    return x, y


def distance_angle(x: float, y: float):
    """Distance (ft) and angle (deg off centerline) to the attacking net."""
    if x is None or y is None:
        return None, None
    dx = GOAL_X - x
    dist = math.hypot(dx, y)
    angle = math.degrees(math.atan2(abs(y), dx)) if dx != 0 else 90.0
    return dist, angle


def danger_zone(x: float, y: float) -> str:
    """High / mid / low danger, approximating the standard 'home-plate' slot.

    High-danger: the home-plate region in front of the net (inner slot).
    Mid: rest of the offensive zone within ~ the faceoff-dot band.
    Low: everything else (points, perimeter, outside O-zone).
    """
    if x is None or y is None:
        return "unknown"
    # Only count shots taken in the attacking end.
    if x < 25:
        return "low"
    absy = abs(y)
    # Home plate: from the goal line region out to ~ the top of the circles,
    # narrowing with distance from the net (crude but recognizable).
    if 69 <= x <= 89 and absy <= 9:
        return "high"
    if 54 <= x < 69 and absy <= 22:
        # slot / inner, widening band
        return "high" if absy <= (9 + (69 - x)) else "mid"
    if x >= 54 and absy <= 30:
        return "mid"
    return "low"
