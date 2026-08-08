# The drawn path and the cost rank controllers differently, and both are right

Prompted by an observation from watching the visualisation: *"visually, stock pid is the best
controller - the metric seems divorced from actual driving performance."*

## The observation has a real basis

Across 205 moving segments (`ALL[5000:5240]`), ranking by cost and ranking by drawn lane deviation
are close to independent:

    corr(cost, open-loop cross-track) = +0.181

## But the conclusion does not survive a wider sample

| controller | mean cost | mean xtrack | wins cost | wins path |
|---|---|---|---|---|
| pid | 139.50 | 3.137 m | 2 | 61 |
| ff_pi_boot | 64.91 | 3.812 m | 55 | 17 |
| ff_pi_tau | 59.25 | 4.045 m | 3 | 2 |
| **cnn_v4** | **56.23** | **2.003 m** | **145** | **125** |

`cnn_v4` wins BOTH. `pid` beats the ff_pi family on path (which is what was seen) but is last on
cost by a factor of 2.5. The impression came from an unrepresentative segment set: the old
visualisation's most dramatic segment, 06585, is one of the ~30% where `pid` genuinely does hold the
tightest line.

## Mechanism: the eye lowpasses, the cost does not

Turning lataccel into a drawn path is a DOUBLE integral, so its sensitivity to error at frequency f
falls as 1/f^2 in amplitude. The cost is a flat-weighted mean square, with jerk weighted 2x per
squared unit. The two therefore disagree whenever controllers differ in WHERE their error sits:

| controller | mean err | rms err | 0-0.1 Hz | 0.1-0.5 | 0.5-1.0 | 1.0-5.0 |
|---|---|---|---|---|---|---|
| pid | 0.0014 | 0.4058 | 0.8% | 44.4% | 25.1% | 29.7% |
| ff_pi_boot | 0.0250 | 0.3130 | 4.5% | 6.7% | 38.6% | 50.2% |
| cnn_v4 | 0.0088 | 0.3004 | 1.1% | 5.6% | 39.8% | 53.6% |

`pid`'s error is larger but **zero-mean chatter**, which a leaky integrator averages away.
The others are smaller but carry a **persistent lag** through sustained corners, which it retains.
On 06585 that inverts the ranking: `pid` shows 0.425 m drawn deviation to `cnn_v4`'s 0.655 m while
costing 52% more (7967 vs 5244). `pid` also moves the wheel the most (mean |dsteer| 0.0260 vs
`ff_pi_tau` 0.0208) and none of that reaches the drawn path.

**A path view structurally cannot show jerk cost, which is half the metric.**

## A wrong turn worth recording

The first diagnosis was that the page dead-reckoned 40 s open-loop, amplifying DC bias
quadratically. That was wrong: `laneOffsets` already used a 3 s leaky double integrator (DC gain
~9.3, so a 0.009 bias displays as 0.085 m). The open-loop cross-track measured in the analysis
script was a quantity the page never drew. Measure the displayed quantity, not a proxy for it.

## Changes made

- `make_drive_data.py`: segments reselected from the 205-segment sweep by steering-activity band,
  taking the median-cost segment of each, plus 06585 retained and explicitly labelled as the
  counterexample. On the four representative segments `cnn_v4` has both the lowest cost and the
  lowest drawn deviation.
- `drive_template.html`: a chatter bar (mean |dsteer| over the last 2 s) under each steering wheel,
  since that is the only place the invisible half of the metric can appear.
