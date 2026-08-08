# Bug bash, 2026-08-08

Full cynic review of this session's work. Three defects found, two of them changing published
numbers. Everything else checked came back clean; the checks that passed are listed too, because a
review that only reports failures gives no sense of what was actually covered.

## 1. Stale trajectory state (RESULT-ALTERING, fixed)

`ff_pi_traj.update` delegated to its parent whenever `w == 0.0`, and the psi/y recursion sat AFTER
that early return. Harmless at a constant w, but `ff_pi_gate` varies w per step and shuts the gate
~97% of the time, so **the trajectory state advanced on only 2.9% of steps** and was stale whenever
the gate reopened. Arm A of `FINDINGS_TRAJ_SELECTIVE.md` was not testing its stated mechanism.

Fixed by making the recursion unconditional; bit-exactness at w=0 now holds by construction
(`0.0*x` is exactly `0.0`) rather than by branching. Re-ran held out, n=400: correcting it made the
arm **worse**, with `lo=0.3` going from +0.719 [-0.083, +1.661] to +2.109 [+0.563, +3.905], now
excluding zero. Conclusion unchanged in direction and strengthened.

Audited the same pattern everywhere: `_awu`, `ff_pi_notch` and `fuzzy` all switch on constant
construction-time flags that never toggle per step, so no state can go stale. Unique to this pair.

## 2. Wrong break-even multiplier (RESULT-ALTERING, corrected)

`FINDINGS_FF_TRAJ.md` reported the `pid` row's break-even jerk weight as "~7x current". That figure
did not follow from its own row: `10.95 / 4.62` = **2.37x**. The 7x was carried over from the
60-segment tuning set (which gives ~5.6x) and printed beside held-out numbers. The row's d(lat) was
also `+11.20` where `50*(2.199-1.980)` is 10.95.

The summary claim "feedforward moved the crossover from roughly 7x to 1.75x" was therefore
overstated. Honest version: **2.4x to 1.75x**. Also recorded: the two rows come from different
held-out samples (n=200 vs n=300), so comparing their exchange rates is indicative, not matched.

## 3. Diverging default constructor (landmine, fixed)

`pid_traj()` with no arguments built a controller at DC gain 9.0 -- the pre-sweep values -- which
diverges, scoring 4,513 to 64,766 against stock `pid`'s ~103. Defaults are now the tuned values,
verified bounded, identity gate re-passed.

## 4. CNN headline contamination: no evidence, but the test is underpowered

`cap_ab.py` trains on `ALL[2000:4000]`, while the cnn headline was quoted on `ALL[:5000]` -- a **40%
overlap**. The docstring asserted no memorisation, but from a reseeding test rather than a direct
measurement.

A raw train-vs-unseen gap cannot settle this, because ranges differ in intrinsic difficulty. Matched
control: `ff_pi_boot`, which saw no training data, over the identical ranges. Medians:

| range | cnn_v4 | ff_pi_boot | ratio |
|---|---|---|---|
| TRAIN `[2000:2150]` | 40.669 | 42.439 | 0.9583 |
| VAL `[4000:4200]` | 43.765 | 46.291 | 0.9454 |
| UNSEEN `[5000:5150]` | 41.809 | 43.806 | 0.9544 |
| UNSEEN `[9000:9150]` | 44.356 | 43.506 | 1.0195 |

difference-in-differences = **-0.0287**. The CNN's edge over the control is ~4.2% on train and ~4.6%
on the nearest unseen range -- essentially the same. Memorising 2000 segments would show far more.

**But this is not a clean pass.** The script's automatic verdict used an arbitrary 0.03 threshold and
cleared it by 0.001. More importantly the **two unseen ranges disagree with each other by 0.065**,
more than double the effect being tested for, so at n=150 between-range variation swamps the signal.
The test is underpowered, not conclusive.

### RESOLVED at n=2000

Rather than argue about the overlap, the headline was recomputed on pristine `ALL[5000:7000]`
(n=2000, never trained, never selected on, never tuned), with matched controls run over the
identical segments (`heldout_headline.py`):

| controller | pristine mean | pristine median | lataccel | jerk | on the 5000 basis | shift |
|---|---|---|---|---|---|---|
| cnn_v4 | 47.366 | 43.218 | 27.097 | 20.269 | 45.32 | +2.04 |
| ff_pi_tau | 51.264 | 45.435 | 32.273 | 18.991 | 49.47 | +1.79 |
| pid | 112.663 | 73.394 | 86.957 | 25.706 | 110.76 | +1.90 |

**difference-in-differences vs ff_pi_tau = +0.25.** The controller that never trained on anything
shifts by nearly the same amount, so ~87% of the 2.04 is that `ALL[5000:7000]` is a harder stretch
of road; contamination accounts for at most ~0.5% of cost. This agrees with the n=150 ratio test
(-0.029) at 13x the sample size, and supersedes it -- the earlier "underpowered" caveat is settled.

`cnn_v4` beats `ff_pi_tau` on pristine data by **-3.897, 95% CI [-4.794, -3.080], winning
1470/2000**. The learned controller's advantage is real on segments it has never seen.

Both bases are now stated in `README.md` and in `controllers/cnn.py`'s header, so the 5000-segment
leaderboard number stays comparable while the overlap is disclosed rather than footnoted.

## Checked and clean

* All four identity gates pass **bit-exactly** after the fixes (ff_pi_traj, ff_pi_gate,
  ff_pi_blend b_fix=0, and ff_pi_blend b_fix=1 against ff_pi_traj w=1).
* **No promoted controller file differs from master** -- every `traj` branch change is additive.
* The sim is **fully deterministic per segment** (4 repeat rollouts identical to 1e-6), so paired
  deltas carry zero Monte Carlo noise, and `process_map` preserves order so pairing aligns.
* `total == 50*lat + jerk` reconciles on **every row of every findings table** checked.
* Parseval conserves to 1.00000. `errdecomp.py` omitted the Nyquist-bin halving; re-running with it
  changes no reported band share at two decimals.
* The page's JS `laneOffsets` and the analysis script's Python agree to **six decimals**, closing the
  gap that produced a wrong diagnosis earlier in the session.
* The `EX` exaggeration appears only in drawing code and its own label, never in quoted metres.
* No mutable default arguments, no module-level shared state across controllers.
* Build guards active; the published page has 0 unsubstituted placeholders and 0 non-ASCII bytes.
* `FINDINGS_VISUAL_VS_COST.md` reconciles exactly against its source log.

## Known limitations -- all since fixed

* `ff_pi_blend` built its trajectory endpoint from `ff_pi_traj` DEFAULTS (tau=3.0) rather than the
  tuned tau=1.0. **Fixed** via `TRAJ_TUNED`. Re-ran: best 45.93 against 44.93 with the untuned
  endpoint -- WORSE, so the handicap was not what made the arm null. The bang-bang degeneracy is
  intrinsic to the one-step model, as suspected.
* `ff_pi_blend` forwarded `kw` to both sub-controllers by double-splatting, so an overlapping key
  raised a duplicate-argument `TypeError`. **Fixed** by merging into one dict with `traj` winning.
* `ff_pi_gate`'s EMA measured error against the raw target while the loop tracks the smoothed
  reference, charging the deliberate cost-optimal smoothing deviation to the tracking side.
  **Fixed**: `ff_pi_traj` records the feedback error it uses (`e_last`) and the gate reads it.
  Re-ran: 41.59 vs parent 41.64, where the previous version gave 41.46 -- the apparent effect
  shrinks to 0.1% with lataccel identical to three decimals on every row. Firing rate is unchanged
  (mean w 0.0124 vs 0.0137), so the correction moved the signal, not the schedule. The gate arm's
  null verdict is weaker still.
