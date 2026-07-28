# The rate-clamp gradient: the defect is real, my explanation of it was wrong

## The claim I had been making

`torch_sim.rollout` hard-clamps the plant's prediction to the rate limit:

```python
pred = torch.clamp(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
```

`torch.clamp` has zero gradient where it binds, so I claimed repeatedly -- in commit messages and in
`FINDINGS_TAIL.md` §13 -- that the policy receives **no gradient signal** at rate-saturated steps,
and drew an analogy to the MPC bug where a hard clamp inside the planning model blinded the
optimiser (worth 2472 -> 6.17 when fixed).

## What measurement says

`grad_probe.py` makes the actions free leaf tensors (initialised from cnn's own actions, so the
trajectory and its saturation pattern are realistic), backprops the benchmark cost to them, and
compares the gradient arriving at saturated vs normal steps:

```
n=32 segments | saturated steps: 9 of 12768 = 0.07%

  st_clamp=False   |grad| SATURATED = 6.719e+01   normal = 9.869e+00   ratio =  6.81
  st_clamp=True    |grad| SATURATED = 1.211e+02   normal = 1.043e+01   ratio = 11.61
```

**The gradient at saturated steps is not zero. It is 6.8x LARGER than at normal steps.**

The error was about graph structure, not about `torch.clamp`. An action does not only affect the
prediction at its own step -- it enters the plant's **20-step input window**, so it also influences
the next 20 predictions. The clamp kills one path out of twenty-one. And saturated steps are
high-error moments, so what does flow through the surviving paths is unusually large.

## What the fix actually does

`st_clamp=True` adds a straight-through estimator: forward stays bit-identical (verified -- costs
match exactly across 8 segments), backward passes gradient through the clamp. It roughly doubles the
gradient at saturated steps (67 -> 121), on **0.07% of steps**. Expected effect on training:
negligible.

## Reconciling with the MPC bug

The MPC failure was real. The difference is *frequency*: that planner saturated almost constantly --
the symptom was a permanent +-0.5 square wave -- so nearly every gradient path was clamped. Here
saturation is rare, so the identical code defect is immaterial.

**The severity of a zero-gradient nonlinearity scales with how often it binds**, which is the thing I
never checked before transferring the analogy from one setting to the other. Worth remembering: a
correct statement about an operator (`clamp` has zero gradient) does not license a conclusion about
a *system* without measuring how much of the system's gradient actually flows through it.

## Status

`st_clamp` is available in `torch_sim.rollout`, defaults to `False` (the exact behaviour the
released checkpoint was trained with), and forward output is unchanged. A training A/B was not run:
the mechanism measurement predicts an effect far below what this metric can resolve, and five prior
cnn training A/Bs at this scale all returned null.
