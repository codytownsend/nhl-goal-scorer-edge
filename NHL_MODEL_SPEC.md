# Porting the "Situational Model" to the NHL — build spec

This is a hand-off spec for building, in a fresh session, an NHL version of a
college-football team-quality model we built (`situational_model/` in the CFB repo).
It describes the **idea**, the **method**, the **hard-won lessons**, and the
**NHL-specific translation** so you can rebuild and test it for hockey. The goal is
a standalone winner/goal-margin predictor, honestly evaluated against the market
(moneyline). It may well underperform the market — that's fine; we want to build and
test it.

---

## 1. The core idea (sport-agnostic)

Build an **opponent-adjusted team-quality rating from event-level play data**, walk
it forward through the season (no leakage), anchor early-season with a **decaying
prior** from last year, map the rating gap to an **expected score margin**, turn that
into a **calibrated win probability**, and compare to the market **only for
evaluation** — never as a model input.

The whole philosophy is: *measure the repeatable, opponent-adjusted quality of each
team from what happens on the ice/field, and predict from that alone.* The market
(and any orthogonal info like player availability) is a benchmark you try to beat,
not an ingredient.

---

## 2. The method (six steps)

1. **Native efficiency metric.** Pick a per-event metric that measures "playing well"
   and is *repeatable* (not luck). In CFB it was **success rate** (did the play stay
   on schedule). NOT points — points/goals come last.
2. **Opponent-adjusted rating.** For each team, solve a rating for offense and defense
   such that `observed_metric ≈ league_avg + off[team] + def[opponent]`. We used a
   weekly iterative/least-squares solve over prior games. This yields how much better
   than average each team makes that metric, adjusted for who they played.
3. **Decaying cross-season prior (the key to early-season sanity).** A team's rating
   each game = empirical-Bayes blend of **last season's final rating (carried ~0.75)**
   and **this season's opponent-adjusted metric**, with weight `K/(K+games_played)`
   (we used K=3). Early in the year it's mostly last season (anchors thin data); by
   midseason it's all current form. This both fixes absurd early predictions and lets
   you predict from game 1.
4. **Rating → expected margin, self-contained.** Fit a **saturating map**
   `margin = A·tanh(rating_gap / s) + c` on historical data (train seasons only).
   It's ~linear for normal gaps (stays calibrated) but bounded at extremes so a
   thin-data blowout rating can't produce an absurd margin. **Do not use the market
   to cap/shrink** — that corrupts the very disagreements you'd bet on.
5. **Calibrated win probability.** `win_prob = Φ(|predicted margin| / σ)`, where σ is
   the model's historical residual std. Verify calibration (an 80% call should win
   ~80%).
6. **Compare to the market.** Convert your predicted margin to your own line; compare
   to the book. Report winner accuracy, calibration, and — honestly — betting results
   with correct break-even math. Then **forward-test live**, because backtested
   "edges" routinely evaporate.

Everything is **walk-forward and leak-free**: a game's prediction uses only games
before it, and the margin map / any tuning is fit on *other* seasons
(leave-one-season-out).

---

## 3. Methodology rules (what makes it rigorous and portable)

- **Reliability gate BEFORE modeling any feature.** Split each team-season's games in
  half, compute the feature on each half, correlate across team-seasons (Spearman-
  Brown to full-season). High = a real, repeatable trait worth modeling; near zero =
  luck, drop it. In CFB this killed fumbles and raw field-goal% as pure noise. Do this
  for every candidate metric first.
- **Leave-one-season-out (LOSO)** for all evaluation and any fitting.
- **Never feed the market into the model.** Market is benchmark-only.
- **Honest betting math.** Spread/puck-line break-even ≈ 52.4% at −110; moneyline
  favorites need a win rate above their (juiced) implied prob. Parlays multiply the
  vig — they only help if legs are genuinely +EV and independent.
- **Assume small-sample "edges" are noise until a live forward test says otherwise.**

---

## 4. Hard-won lessons from the CFB build (save yourself the dead ends)

- **Situational granularity collapses into overall quality.** We built a 72-cell
  situational model (by down/distance/field-zone/score); a *flat* overall
  opponent-adjusted rating **beat it** (0.700 vs 0.665 winner accuracy). Fine-grained
  per-situation rates are too noisy and just add variance. **Start simple** (overall
  efficiency); only add situational splits if they pass the reliability gate AND beat
  flat out-of-sample. They probably won't.
- **The map's shape doesn't change *who* you pick** — winner = sign of the rating gap.
  Nonlinearity only tames magnitudes (useful for not producing crazy lines).
- **The decaying prior is worth it** — it raised full-slate accuracy and fixed
  early-season nonsense.
- **Most fashionable decompositions add nothing** beyond team quality (we tested
  "style/matchup", turnovers, special-teams/kicking — all collapsed into quality or
  were too noisy). Expect the same; test them cheaply and move on.
- **The market is very hard to beat.** Our model hit ~72% straight-up vs the market's
  ~75%, matching it only late in the season. The genuine value was (a) standalone/
  unlined prediction and (b) narrow, unconfirmed betting angles worth a live test —
  the best being a **moneyline** angle (small favorites we strongly backed), not the
  spread.
- **The one clearly-orthogonal signal was player availability** (QB in/out). In NHL
  this is even bigger — see goalies below.

---

## 5. CFB → NHL translation

| CFB concept | NHL analogue |
|---|---|
| play (run/pass) | on-ice event (shot attempt, etc.), at **5v5** |
| success rate (native metric) | **5v5 expected-goals share (xGF%)**; fallback **Corsi%** (shot-attempt share) — needs no location model |
| points / point margin | goals / **goal differential** |
| opponent-adjusted efficiency rating | opponent-adjusted **xG-differential** rating |
| down/distance/field/score "situations" | **strength state** (5v5 / power play / penalty kill / 4v4 / empty-net) and **score state** (lead/trail/tied) |
| home-field advantage | **home-ice advantage** (home teams win ~55%) |
| **QB availability** (orthogonal edge) | **STARTING GOALIE** + goalie quality — the single biggest input; treat as first-class, not an add-on |
| rest / travel | **back-to-backs** (2nd night, esp. backup goalie), travel, altitude (Denver) |
| special teams / kicking (tested, minor) | **power play / penalty kill** — test reliability first (PP% is noisy, small sample) |
| Vegas spread | **moneyline** (primary in NHL) + **puck line ±1.5** |
| ~72% winner acc; market ~75% | **expect ~56–60% winner acc; market ~57–60%** (see below) |

### The single most important NHL-specific point: variance
Hockey is *far* more random than football. Games are low-scoring (~3 goals/side) and
often decided by one goal or a bounce, so even elite models and the sharp market hit
only **~57–62%** on straight-up winners. Set expectations there: a "good" NHL team
model is ~58% straight-up, not 72%. Much of a single game is goalie + luck. This is
the biggest reason it may "not work" as a winner predictor — and why goalie modeling
matters so much.

---

## 6. NHL build plan

**Data (NHL API).** Use the official NHL API (currently `api-web.nhle.com`; verify
current endpoints — they change). You need, per game, **play-by-play events** with
event type, **coordinates**, **strength/situation code**, and score/time; plus the
**schedule**, **final scores**, and **starting goalies** (announced pregame). Season
is 82 games × 32 teams → lots of sample. Consider also grabbing pre-computed xG from a
public source (MoneyPuck / Natural Stat Trick) to sanity-check your own xG.

**Native metric.** Start with **5v5 Corsi% (shot-attempt share)** — trivial to compute
(count shot attempts for/against at 5v5), no location model. Then upgrade to **xGF%**:
build a simple expected-goals model = logistic of goal ~ shot distance/angle/type from
the event coordinates (or borrow public xG). xG is the better, more predictive metric;
Corsi is the robust starting point.

**Rating.** Opponent-adjust the 5v5 metric exactly as in CFB (offense = generating
xG/attempts, defense = suppressing them), walk-forward, with the **decaying prior**
(seed each season from last season's rating × ~0.75; blend K/(K+games)). Hockey teams
turn over less roster-to-roster than CFB but goalies/coaches change — 0.75 is a
reasonable start; tune it.

**Map to goals.** Fit `goal_margin = A·tanh(rating_gap/s) + c` (train seasons only).
Goal differentials are small (mostly −3..+3), so A is small and the tanh saturates
quickly — good. Add **home ice** via the intercept, and a **rest/back-to-back** term.

**Win probability.** Either `Φ(margin/σ)` with the (large) hockey σ, or model goals as
**two Poisson / a Skellam** distribution (expected goals-for each team) and compute
P(win) incl. a fixed OT/shootout coin-flip for ties. The Poisson route is more natural
for hockey and gives you totals (over/under) for free.

**Goalies (do this early, it's central).** Get the **confirmed starting goalie** each
game and a leak-free **goalie-quality rating** (goals saved above expected / GSAx, or
save% above expected, shrunk toward league, carried across seasons). Adjust the game's
expected goals-against by the starter's quality vs a league-average goalie. A backup
starting (or the 2nd night of a back-to-back) is a large, orthogonal swing — the NHL
equivalent of the QB-out signal, and probably your best shot at a real edge.

**Special teams.** Test PP%/PK reliability with the split-half gate before modeling. If
weak (likely), fold only a shrunk PP/PK rate into expected goals; don't over-build.

---

## 7. Evaluation & forward test

- **LOSO by season**: winner accuracy, log-loss, calibration curve.
- **Vs the market**: winner accuracy head-to-head on the same games; who's right on
  disagreements; **moneyline** ROI with correct (juiced) break-even; puck-line ATS.
- **Calibration**: bucket win_prob, check observed win rate matches.
- **Live forward test**: log predictions for upcoming games *before* they're played
  (pull the day-of starting goalie), grade weekly, track moneyline/puck-line P/L vs
  real prices. This is the only trustworthy verdict.

Success looks like: a calibrated ~57–60% straight-up model, competitive with the
market late in the season, with goalie/rest as the orthogonal levers. "Failure"
(doesn't beat the market) is the expected and acceptable outcome — you'll still have a
sound standalone predictor and a clean framework to test angles.

---

## 8. Suggested build order (milestones)

1. Pull one season of play-by-play + schedule + final scores from the NHL API; get the
   data shapes right. Compute 5v5 **Corsi%** per team per game.
2. Reliability gate: split-half Corsi%, xGF% (if built), PDO (expect noise), PP/PK.
3. Opponent-adjusted Corsi/xG rating, walk-forward, one season; sanity-check ratings
   vs standings.
4. Add the **decaying prior** across ≥3 seasons; LOSO winner accuracy.
5. Map to goal margin (tanh) + home ice; calibrate win_prob; compare to moneyline.
6. Add **goalie** starter + quality adjustment; re-evaluate (expect the biggest gain).
7. Add rest/back-to-back. Betting eval (ML, puck line) with honest break-even.
8. Wire a weekly "predict upcoming games" script (pull day-of goalies) + a forward-test
   ledger. Mirror `situational_model/predict_week.py` + `forward_2026.py`.

Reference implementation to imitate (structure, walk-forward, decaying prior, tanh
map, calibration, forward test): the CFB `situational_model/` — `rating_decay.py`,
`forward_2026.py`, `predict_week.py`, `carryover_test.py`, and `FINDINGS.md`/`README.md`.
