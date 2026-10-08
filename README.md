# nhlsit — situational NHL team comparison

Compare how teams play **by situation** — *where* on the ice, *when* (period,
score state), and *at what strength* (5v5 / PP / PK / 4v4 / 3v3 / empty net) —
built straight from NHL play-by-play. This is a **descriptive / scouting** tool
(profile and compare teams), separate from the win-prediction model in
`NHL_MODEL_SPEC.md`.

## What it measures

Every **shot attempt** (Corsi event: goal, shot-on-goal, missed-shot,
blocked-shot) is tagged with:

- **Strength state** — from `situationCode` (`5v5`, `PP`, `PK`, `4v4`, `3v3`,
  empty-net).
- **Score state** — attacking team's lead/tie/trail at the moment of the shot
  (`trail_2+`, `trail_1`, `tied`, `lead_1`, `lead_2+`).
- **When** — period (`P1/P2/P3/OT`) and game seconds.
- **Where** — coordinates normalized so the attacking net is always at +89;
  distance, angle, and a **danger zone** (high / mid / low, approximating the
  standard home-plate slot).
- **xG** — a simple logistic expected-goals model (distance, angle, shot type),
  fit from the data itself. Sanity-grade, not state-of-the-art.

Headline metrics (for **and** against, per team): `CF%` (shot-attempt share),
`xGF%`, `xGF/g`, `xGA/g`, `HDCF%` (high-danger share), `GF/g`, `GA/g`, `Sh%`.

## Setup

Needs `requests`, `pandas`, `numpy`, `pyarrow`. Raw API responses and the built
event table cache under `data/`.

## Usage

```bash
# 1) Build one season's tidy event table (cached; --limit N for a quick test)
python -m scripts.build_events 20232024

# 2) Compare / profile / rank
python -m scripts.compare_teams 20232024 --a TOR --b BOS --strength 5v5
python -m scripts.compare_teams 20232024 --a EDM --by score_state --strength PP
python -m scripts.compare_teams 20232024 --table xGF% --strength 5v5 --period P3
```

Situation flags combine freely: `--strength`, `--score`, `--period`, `--danger`.

### In code

```python
import pandas as pd
from nhlsit import compare as C
ev = pd.read_parquet("data/events_20232024.parquet")

C.compare(ev, "COL", "VGK", situation={"strength_bucket": "5v5", "score_state": "tied"})
C.team_profile(ev, "FLA", by="period_label")
C.league_table(ev, "HDCF%", situation={"strength_bucket": "PP"})
C.danger_mix(ev, "CAR", side="defense", situation={"strength_bucket": "5v5"})
```

## Win predictor (walk-forward, leak-free)

Built on top of the same events. Opponent-adjusted 5v5 xG ratings
(`home_xg_diff ~ home_adv + rating[home] - rating[away]`, ridge-shrunk toward
last season's rating × 0.75 → the decaying prior), a saturating
`margin = A·tanh(gap/s)+c` map, and `P(home win) = Φ(margin/σ)`.

```bash
# leave-one-season-out evaluation (accuracy, log-loss, calibration)
python -m scripts.evaluate 20212022 20222023 20232024

# predict a single matchup before puck drop
python -m scripts.predict_game --home EDM --away CGY
```

**LOSO results (3,936 test games, 2021–24; seeded by a 2020-21 prior),
tuned `lam=3, carry=0.6`:** winner accuracy **61.1%** (vs 53.4%
always-pick-home), log-loss 0.656 (baseline 0.691), well calibrated. That's in
the ~57–60% band the spec predicted — a sound *team-quality-only* model.

### Levers (all walk-forward / leak-free)

| Model | Winner acc | Log-loss |
|---|---|---|
| base (5v5 xG rating) | 59.9% | 0.6641 |
| + starting goalie (GSAx) | 60.1% | 0.6615 |
| + rest / back-to-back | 60.1% | 0.6620 |
| + special teams (PP/PK) | 60.7% | 0.6599 |
| **full (all)** | **61.1%** | **0.6563** |

- **Goalie** (`goalie.py`): starter = who faced the most on-goal shots; quality =
  GSAx per shot faced, shrunk toward league and carried across seasons.
- **Rest** (`rest.py`): capped rest differential; back-to-back = 1 day.
- **Special teams** (`special.py`): net PP/PK xG quality. Passes the split-half
  reliability gate (PP xGF 0.78, PK xGA 0.60 — xG rates repeat even though raw
  PP% doesn't), and is the biggest single lever here.
- **xG** now includes **rebound/rush** context (rebounds convert ~2–3× baseline).
- **Prior season** (2020-21) seeds ratings so 2021-22 isn't a cold start — that
  season alone rose 59.7% → 61.5%.
- **Skellam win-prob** (`predict.skellam_win_prob`, `--winprob skellam`) available
  for **totals**, but the Normal-CDF is better calibrated for winner probability.
- **Tuning** (`scripts.tune`): the (lam, carry) surface is flat; defaults are
  near-optimal.

### Market comparison (benchmark only — never a model input)

`market.py` + `scripts/market_eval.py` compute, given moneyline odds:
straight-up accuracy vs the book, who's right on disagreements, and flat-stake
ROI on the model's +EV side with correct juiced break-even.

```bash
python -m scripts.market_eval path/to/odds.csv   # CSV: date,home,away,home_ml,away_ml
```

Odds are **not bundled** (no reliable free historical feed). Supply your own CSV
(see `data/odds_template.csv`), or pull live upcoming lines for the forward test
via `nhlsit.market.fetch_live_odds()` (The Odds API free tier; set
`THE_ODDS_API_KEY`). Until real odds are loaded, "can it beat the market" is
untested — as the spec stresses, the live forward test is the only real verdict.

Modules: `rating.py`, `goalie.py`, `rest.py`, `predict.py`, `market.py`;
scripts `evaluate.py`, `predict_game.py`, `market_eval.py`.

## Layout

```
nhlsit/
  api.py         # HTTP + on-disk JSON cache (NHL web API)
  fetch.py       # season -> game IDs (via club schedules)
  parse.py       # play-by-play -> tidy shot-attempt table
  situations.py  # strength / score / zone / coordinate logic
  xg.py          # simple logistic expected-goals model
  compare.py     # aggregation, team-vs-team, league tables
scripts/
  build_events.py    # build data/events_<season>.parquet
  compare_teams.py   # CLI for comparisons
```

## Notes / caveats

- Shares and per-game rates are used throughout (no time-on-ice needed), so
  PP/PK numbers are per game, not per-60.
- The danger-zone polygon is a recognizable approximation of the home-plate
  area, not an official definition.
- `eventOwnerTeamId` is the **shooting** team for all shot types (including
  blocked shots — the block coordinate sits in the attacker's offensive zone).
- xG here is for description; upgrade it (rebound/rush/pre-shot movement) only
  if a use case earns it.
```
