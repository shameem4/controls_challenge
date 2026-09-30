# Neural MPC: policy proposes, planning refines — and refinement makes it worse

*Recorded on branch `neural-mpc`, last updated 2026-07-28. Headline numbers quoted here are those current at the time; `master` has since moved on — see the README for the figures that stand today.*

Follows the diagnosis in `FINDINGS_TAIL.md` (branch `tail-analysis`), which separated three
obstacles for the model-based route:

* **model fidelity — solved.** The exact model is the network itself. Fitted surrogates are capped:
  linear ARX plateaus at ~1.35x the noise floor and ARX(3,3) sits at 1.98x, statistically the same
  as the model that produced a ~78 controller.
* **expected-mode exploitation — solved.** Planning against the deterministic mean gave planned ~50
  / realised 432. Averaging over K sampled scenarios inverts that: planned now exceeds realised.
* **the inner solver — the remaining bottleneck.** Adam from a feedforward warm start improved on
  its own initialisation by ~8%.

## The idea

Attack the solver where it costs least: replace the weak feedforward initialisation with the
**learned policy**. `cnn` (47.87) is a far better proposal than inverse-plant feedforward (~59), so
the optimiser starts near a good solution and only has to correct it. That is the propose-and-refine
structure of neural MPC — amortised policy for the coarse answer, short online optimisation for the
residual.

## Result: refinement destroys the proposal

`neural_mpc.py`, n=12, H=12, iters=3, K=3, identical segments and code, **identical RNG
consumption**, differing only in refinement step size:

```
  LR=1e-6   ->  45.935      refinement effectively off
  LR=1e-3   ->  45.563      -0.37, inside noise at n=12
  LR=3e-2   -> 229.051      5x worse, worse on 12 of 12 segments
```

Neutral where it is too small to matter; catastrophic once large enough to move anything. There is
no useful step size.

### Getting the control right mattered

The first comparison used an `iters=0` arm as the control, on the assumption that a fixed seed gave
both arms the same noise. **That was wrong**: the refinement arm consumes RNG during Gumbel
sampling, desynchronising the executed plant noise. The correct control is `LR=1e-6`, which consumes
*identical* RNG and differs only in step size. That also rules out a wiring bug — at negligible LR
the harness performs normally (45.9), so the degradation is the refinement itself.

### Mechanism: the optimiser has its own attractor

Comparing across warm starts is what identifies the cause:

```
  from feedforward start (~462):  refinement IMPROVES to ~426
  from cnn start          (~46):  refinement DEGRADES  to ~229
```

Both land in the same 230-460 band. The inner optimisation is not refining a proposal — it converges
toward its own (bad) optimum regardless of initialisation, so a better proposal simply gives it more
to destroy.

The likely cause is that the horizon objective is **myopic**: 12 steps with no terminal value. The
optimiser honestly minimises what it is given and pays for it beyond the horizon. `cnn`, trained
end-to-end over full episodes, implicitly carries the cost-to-go that this objective omits. Note the
degradation is mostly jerk (125.6 of 229.1) — consistent with an optimiser willing to end its
horizon in a state that is expensive to leave.

## What this closes, and what it does not

**Closed:** propose-and-refine neural MPC using *online gradient* refinement. More iterations or a
better-tuned step size will not fix an objective that is wrong.

**Not closed, and now precisely specified:** the missing piece is a **terminal value function**, not
a better optimiser. A cost-to-go network used as terminal cost on a short horizon would remove the
myopia this experiment exposed. That is a scoped project, not an afternoon: train a value net on
realised cost-to-go, verify it on held-out segments, then re-run this harness with it.

Also still open: a **learned correction** (train a network offline to map proposal -> refined
action), which avoids online optimisation entirely.

## Cost note

This harness is a bound, not a submittable controller: seconds per step, and it calls the plant as
an oracle. Everything here was measured at n=12, adequate for a 5x effect and useless for a 1-point
one.
