# Gain prior on the CNN, and the control run that retracted the `cnn_dual` promotion

Two results, one intended and one not.

## 1. The intended experiment: measured gain prior on the CNN — NULL

The bootstrap that was worth −38% on the stock PID and −1.08 on `ff_pi_rl2` is not anticipation; it
is a **measured physics prior**, `G(v)` from `gain_fit.npy` (±9%). The CNN has to discover its gain
schedule with `self.film = nn.Linear(1, 2)` — two parameters standing in for a fitted quadratic — so
there was a real representational gap to close.

Implementation: cfg flag `G` divides the preview window by `G(v)` inside `AblNet.forward`, so the
conv sees the window in **steering units**. `film` is retained as a learned correction on top, so a
null reads as "the CNN already learned the schedule" rather than "we broke the net".

Verified before spending GPU time: `PM` path untouched, `plant_gain` exact vs `np.polyval`
(max err 0.0e+00), **zero** train/serve skew between the numpy eval and torch training paths,
identical parameter count (11243 both, so not extra capacity), and `G` demonstrably changes the
function.

Both runs: `SEED=0`, `TBPTT=60 TRAIN_N=2000`, 1000 iters, `gumbel_soft`. The shared seed makes this a
**paired** A/B — same init, same batch order, the `G` flag the only difference.

    torch surrogate best        PMG 41.12   PM 41.81      (-0.69 to PMG)
    real-sim selection split    PMG 46.166  PM 46.765     (-0.60 to PMG)
    pristine ALL[4200:5000]     PMG 50.868  PM 50.720     (+0.15, null)
    headline ALL[:5000]         PMG 47.162  PM 46.911     (+0.25, null)

    PMG vs PM, pristine   mean +0.149 [-1.33,+1.41]  median -0.0712  better 421/800
    PMG vs PM, headline   mean +0.251 [-0.10,+0.60]  median -0.0947  better 2695/5000

**Null.** Note the shape of the failure: the gain prior led on the surrogate AND on the selection
split, and the lead vanished on both untouched splits. That is winner's curse, and it is a reminder
that the selection split cannot be used to size an effect — only to pick within a family.

The bootstrap's payoff tracks how much physics the host already has, monotonically to zero:

| host | what it already had | bootstrap gain |
|---|---|---|
| stock PID | no feedforward at all | −38% |
| `ff_pi_rl2` | explicit ff + `gain_fit.npy` schedule | −1.08 |
| `cnn` | learned ff + learned schedule, trained on the real cost | **null** |

Selection was restricted to iters >= 500, where both runs had converged (last five checkpoints:
PMG 41.86–43.18, PM 41.81–42.98). Stated because it is a cap on the search.

## 2. The unintended result: `cnn_dual`'s gain was training run variance

`cnn_dual.pt` was promoted earlier in this session on `47.872 -> 46.894` against `cnn_PM.pt`. The
`PM` control run of this experiment is the comparison that promotion never had: a **fresh
from-scratch run of the same architecture and recipe**.

    headline ALL[:5000]
      cnn_dual (BC->PO, shipped)   46.894
      fresh PM control             46.911
      cnn_PM (previous default)    47.872

      cnn_dual vs cnn_PM   mean -0.978 [-1.51,-0.54]  median -0.2022  better 2910/5000
      cnn_dual vs FRESH    mean -0.016 [-0.61,+0.39]  median +0.0047  better 2490/5000   <- null
      fresh PM vs cnn_PM   mean -0.961 [-1.41,-0.54]  median -0.1998  better 2889/5000

    pristine ALL[4200:5000]
      cnn_dual vs cnn_PM   mean -2.997 [-6.13,-0.79]  median -0.2144  better 477/800
      cnn_dual vs FRESH    mean -1.409 [-4.57,+0.38]  median -0.0020  better 401/800     <- null
      fresh PM vs cnn_PM   mean -1.588 [-3.23,-0.27]  median -0.1811  better 452/800

A plain rerun reproduces essentially the **entire** −0.978. Against a matched control, `cnn_dual` is
a coin flip on both splits (2490/5000 and 401/800, medians +0.005 and −0.002).

**Behaviour cloning followed by policy optimisation gives nothing over policy optimisation from
scratch.** `cnn_PM.pt` was simply a below-average training run, and the entire "BC lands in a
different basin" line of investigation was measuring run-to-run variance. This is consistent with
the two earlier negative signals that were noted but not acted on: the geometry probe refuted the
flat-minimum account, and iterating BC->PO did not compound.

This also resolves an entry in the README's Unknowns section rather than leaving it open.

### The methodological error, stated plainly

The promotion compared a new artifact against an **old** artifact instead of against a **matched
control**. Both gates that were applied — bootstrap CI and a sign test — were correctly computed and
both passed; they simply answered "is `cnn_dual` better than `cnn_PM`?" when the question that
matters is "is the BC->PO *schedule* better than the plain one?". Statistical rigour on the wrong
comparison is still the wrong answer.

Run-to-run training variance on this architecture is **~1.0 point on the headline and ~1.6 on an
800-segment split** — larger than most effects chased in this project. Any future training-side
claim needs a fresh matched control, not a comparison against whatever was shipped last.
