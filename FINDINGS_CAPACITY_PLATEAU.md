# Capacity, retested at a plateau — and where the training-length gains stop

## Why retest

`15fa575` compared 11k and 89k parameters at a fixed **400 iterations** and concluded capacity is not
the limit. That budget is now known to be far from convergence (`FINDINGS_GRAD_ARM.md`), and the same
commit observed the two widths have different learning curves — "at iteration 50 the wide model was
~2x better … that was learning SPEED, not final quality". Comparing different curves at an arbitrary
early point says nothing about their asymptotes, so the conclusion needed re-testing even though it
was the right call at the time.

The objection that prompted this: *if more training helps a small model, a bigger model might need
even more training before its advantage appears.* That is a sound argument and it deserved a test.

## Design

Train each width to **its own plateau**, not to matched compute — a bigger model legitimately needs
more compute, and forcing an equal budget would rebuild the original bias. Stopping rule: no new best
for 16 consecutive validations (1200 iterations).

A convenient fact made this cheap: **`ch=96` costs the same per iteration as `ch=32`** (11.24 vs
11.31 s/iter). The 1M-parameter plant rollout dominates, not the policy net, so the capacity A/B
needs no compute matching at all.

Both arms: from scratch (`wide96`) or resumed (`base4`), same pool, curriculum, `ACC=4`, checkpoints
selected on the real numpy sim over `ALL[4200:4700]`.

## The patience rule had to be doubled

`wide96` first hit the plateau rule at iteration 2025 with a best of 45.98. That stop was **wrong** —
`base4` had twice recovered from stale=6/8 to set new bests, so patience 8 is marginal on a
200-segment validation metric this noisy. Resumed with patience 16, `wide96` went on to 44.82 by
iteration 3450, well past its "plateau". Patience 16 was then applied to both arms for symmetry.

Recording this because it is the same failure as the original experiment, one level up: a stopping
rule that is too aggressive manufactures a false ceiling.

## Result: capacity still does not help

Real sim, best checkpoint of each arm selected on `ALL[4200:4700]`:

| | best iteration | selection mean |
|---|---|---|
| `base4` 11,243 params | 5325 | **44.288** |
| `wide96` 89,003 params | 3825 | 44.571 |

| split | `wide96` 89k − `base4` 11k | median Δ | better |
|---|---|---|---|
| `ALL[5000:6000]` n=1000 | −0.044 [−1.81, +1.28] | +0.097 | 465/1000 (z=−2.2) |
| `ALL[:5000]` n=5000 | **+0.450 [+0.12, +0.77]** | +0.150 | 2237/5000 (z=−7.4) |

Tied on one split, **worse** on the headline with a CI excluding zero. Eight times the parameters
does not reach a better asymptote.

What it *does* buy is speed: `wide96` reached its best at iteration 3825 against `base4`'s 5325, and
it was ahead on a per-iteration basis through the middle of training. That is the same "learning
speed, not final quality" pattern the original commit named — now confirmed at a plateau instead of
assumed from an early budget.

So the original conclusion stands, but its evidence was luckier than it looked: both arms were
undertrained when it ran, and it happened to get the right answer for a reason that did not hold.

## Training length: one more real gain, then diminishing

| | headline `ALL[:5000]` | median | p99 |
|---|---|---|---|
| **`cnn_v4` = `base4` it5325** | **45.322** | **43.22** | 144.3 |
| `cnn_v3` = `base4` it2625 | 45.744 | 43.98 | 132.3 |
| `wide96` it3825 | 45.772 | 43.47 | 140.3 |
| `cnn_v2` (400 iters) | 46.911 | 44.14 | 148.2 |

`cnn_v4` vs `cnn_v3`: **−0.422 [−0.86, −0.02]** on the headline, median −0.518, better on 3295/5000
(sign z=+22.5). Promoted. Note p99 moves the wrong way (144.3 vs 132.3) — the gain is in the bulk,
not the tail.

The trajectory of the training-length gains is the reason to stop here:

    400 iters   -> 46.911
    2700 iters  -> 45.744   (-1.167 for 2300 iterations)
    5325 iters  -> 45.322   (-0.422 for 2625 iterations)

Halving returns per equal increment, at ~10 hours per increment on this hardware. `base4` ran to 7200
iterations and produced nothing better than its 5325 checkpoint over the last 1875. The
training-length lever is now spent.

## Status of the cnn line

Reopened by `FINDINGS_GRAD_ARM.md`, and now closed again on its own terms:

* the ~48 ceiling was the iteration budget — real, and worth 1.6 points total (46.911 → 45.322);
* capacity is not the limit, retested properly at a plateau;
* training length has reached diminishing returns.

The eleven earlier negatives are unaffected — each was A/B'd against a matched control at its own
budget. The remaining gap to the ~36 noise-free frontier still needs a different method, which is
what the README concluded before this detour.
