# A seed-aware optimum as the per-segment difficulty measure

## Why the analytic floors were not enough

Every floor used before this was analytic: a causal noise floor plus the Tikhonov trajectory optimum,
combined as `floor = sigma^2 * 26614 + J*`. Two problems, both now measured:

1. **The noise term was a population constant.** 31.24 is `E[sigma^2] * 26614` averaged over segments.
   Per segment, `segfloor.py` measures the plant's exact conditional variance
   (`sum p_i b_i^2 - (sum p_i b_i)^2`) and finds the floor ranges from **6.26 (p10) to 52.15 (p90)**.
   The mean, 29.99, reproduces the published 31.24 as a sanity check. The estimate is stable across
   controllers: sigma^2 measured under `cnn_v4` vs `ff_pi_tau` correlates **0.951**, median ratio 0.990.

2. **Additivity was never valid.** A seed-aware plan scores **below** `noise_floor + J*` on 68 of 128
   segments. Knowing the noise draw lets you pre-compensate for it, so the two costs do not add. The
   additive floor is still an excellent difficulty *ranking* -- `corr(log opt, log floor_add) = +0.975`
   -- but it is not a bound.

## The instrument

`steer_opt.py` optimises the action sequence per segment against that segment's fixed noise
realisation. It is a seed exploit by construction and is **not submittable**; it is a measuring device.
It replaces `steer_lookup.py`, which had two defects that made it unusable per segment:

* **Batch-mean acceptance.** A sweep that helped the average was kept even on segments it hurt. On 128
  segments this produced an "oracle" *worse* than the warm start on an entire quartile (42.99 -> 50.13),
  which reads as "the controller already beats the oracle there" and is an artefact.
* **Sweeps that could not differ.** The sweep is deterministic and a rejected sweep restores the prior
  plan, so sweeps 2..N replay sweep 1 exactly — observed as 42.570 four times in a row. Only one sweep
  ever ran.

Fixed: per-segment acceptance and per-segment best tracking; annealed step size (0.6^sweep) with a
shuffled 75% subset of positions per sweep. Result on the same 128 segments:

    steer_lookup   50.301 -> 42.394, then stuck (4 identical rejected sweeps)
    steer_opt      50.301 -> 38.249 mean / 31.782 median, 128/128 segments improved

Still an upper bound on achievable cost: the best public exploits reach ~6, so this is "the best plan
we can find", not the optimum.

## Result

128 pristine segments, `cnn_v4` and the optimiser in the *same* rollout under the same seed:

| | mean | median |
|---|---|---|
| `cnn_v4` | 50.30 | 40.44 |
| seed-aware optimum | 38.25 | 31.79 |
| additive floor | 36.38 | 34.90 |

Gap 12.05 mean, 7.39 median; the optimum wins on **128/128**.

| quartile by cnn/opt ratio | n | `cnn_v4` | optimum | ratio | median J* | mean sigma^2 |
|---|---|---|---|---|---|---|
| already near-optimal | 32 | 48.50 | 43.26 | 1.124 | 4.77 | 0.00129 |
| 2nd | 32 | 43.44 | 36.41 | 1.190 | 2.60 | 0.00109 |
| 3rd | 32 | 48.52 | 38.53 | 1.256 | 2.30 | 0.00123 |
| furthest from optimal | 32 | 60.75 | 34.79 | 1.547 | 1.50 | 0.00065 |

`cnn_v4` is within 25% of the cheating optimum on 58% of segments, within 10% on 6%, within 2% on none.

## The finding, and the trap in it

The controller is relatively **closest** to the seed-aware optimum on the hardest, noisiest segments
and **furthest** on the easy, quiet ones. That is the opposite of the intuition that hard segments are
where the controller falls short.

It is also mostly not actionable, for a structural reason. On a quiet segment the optimiser's advantage
comes almost entirely from pre-compensating a noise draw it can see; a causal controller cannot do that
at all. On a high-J* segment the cost is dominated by trajectory shape, which a causal controller *can*
attack — and there the gap is smallest, meaning the CNN already captures most of what is available.

So the gap to a seed-aware optimum is an **upper bound** on recoverable cost, not an estimate, and it is
largest exactly where it is least recoverable. Ranking segments by it would point optimisation at the
segments where a causal controller has the least to gain — the same trap the constant-floor banding fell
into, one level up.

## Methodological note

The torch rollout and the numpy simulator draw different random numbers, so the same policy lands on a
different noise realisation in each: `cnn_v4` scores 53.82 in the numpy sim and 50.30 in the torch
rollout on these segments (corr 0.986, mean |diff| 11.9). Any per-segment comparison must be made
inside one of them, not across. An earlier version of this analysis compared the optimiser's torch cost
against the CNN's numpy cost and was invalid for that reason.
