"""Classical system identification: open-loop step responses across operating conditions.

A designed experiment rather than observational fitting. Conditions are SYNTHESISED -- constant
v_ego, roll, a_ego and operating lataccel -- so each is an independent variable instead of a
correlated property of whatever segments happened to be picked. The plant samples, so every row of a
batch is an independent noise realisation and a batch gives the ensemble mean response directly.

Protocol per condition, mirroring a bench step test:
  1. hold steer at u0 until the lataccel settles      (SETTLE steps)
  2. step the command to u0 + du
  3. record the mean response over RESP steps across NREP noise realisations

Extracted per condition: DC gain dc/du, dead time, t10/t50/t90, overshoot, and -- by sweeping du --
whether superposition holds at all, which is the assumption every linear design in this project rests
on (ff_pi's inverse-plant feedforward, the DMC teacher, the Tikhonov reference).

Known from earlier, piecemeal work, for cross-checking: G(v) ~ 0.0093*v + 1.34 (measured 1.43-1.67
rising with speed), impulse response H = [0, .02, .08, .22, .38, .30] normalised to unit DC gain, and
step response t10/t50/t90 = 400/500/700 ms at low speed vs 300/400/500 ms at high speed.

Usage: python sysid.py [tag]     env: NREP, SETTLE, RESP
"""
import sys, os, numpy as np, torch
from pathlib import Path
from torch_sim import Plant
from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX, MAX_ACC_DELTA, STEER_RANGE, DEL_T

DEV = 'cuda'
NREP = int(os.environ.get('NREP', 96))       # independent noise realisations per condition
SETTLE = int(os.environ.get('SETTLE', 150))  # steps held at u0 before the step (see NULL GATE below)
RESP = int(os.environ.get('RESP', 40))       # steps recorded after the step


def synth(v, roll, a, c0, u0, n, T):
    """n identical constant-condition 'segments'. c0 is the pre-step lataccel the warmup forces."""
    return [dict(roll=np.full(T, roll, np.float64), v=np.full(T, v, np.float64),
                 a=np.full(T, a, np.float64), target=np.full(T, c0, np.float64),
                 steer=np.full(T, u0, np.float64)) for _ in range(n)]


class Rig:
    """Minimal batched stepper for constant-condition tests (no segment data, no preview)."""

    def __init__(self, plant, segs, T):
        self.p, self.T = plant, T
        st = lambda k: torch.tensor(np.stack([s[k][:T] for s in segs]), dtype=torch.float32, device=DEV)
        self.roll, self.v, self.a = st('roll'), st('v'), st('a')
        self.target, self.steer0 = st('target'), st('steer')
        self.lat = [self.target[:, i] for i in range(CONTEXT_LENGTH)]
        self.act = [self.steer0[:, i] for i in range(CONTEXT_LENGTH)]
        self.cur = self.lat[-1]
        self.t = CONTEXT_LENGTH

    @torch.no_grad()
    def step(self, u):
        t = self.t
        u = u.clamp(*STEER_RANGE)
        if t < CONTROL_START_IDX:
            u = self.steer0[:, t]
        self.act.append(u)
        s = torch.stack([torch.stack(self.act[-CONTEXT_LENGTH:], 1),
                         torch.stack([self.roll[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1),
                         torch.stack([self.v[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1),
                         torch.stack([self.a[:, i] for i in range(t - CONTEXT_LENGTH + 1, t + 1)], 1)], -1)
        past = torch.stack(self.lat[-CONTEXT_LENGTH:], 1)
        pred = self.p.step(s, self.p.tokenize(past), mode='sample')
        pred = torch.clamp(pred, self.cur - MAX_ACC_DELTA, self.cur + MAX_ACC_DELTA)
        self.cur = torch.where(torch.tensor(t >= CONTROL_START_IDX, device=DEV), pred, self.target[:, t])
        self.lat.append(self.cur)
        self.t += 1
        return self.cur


@torch.no_grad()
def step_test(plant, v, roll, a, c0, u0, du, seed=0):
    """Mean lataccel response to a step of du in the steer command."""
    torch.manual_seed(seed)
    T = CONTROL_START_IDX + SETTLE + RESP + 5
    segs = synth(v, roll, a, c0, u0, NREP, T)
    R = Rig(plant, segs, T)
    U0 = torch.full((NREP,), u0, device=DEV)
    while R.t < CONTROL_START_IDX + SETTLE:
        R.step(U0)
    pre = R.cur.mean().item()
    U1 = torch.full((NREP,), u0 + du, device=DEV)
    resp = [R.step(U1).mean().item() for _ in range(RESP)]
    return pre, np.array(resp)


def metrics(pre, resp, du):
    """DC gain, dead time and rise times from a mean step response."""
    final = resp[-8:].mean()
    d = final - pre
    if abs(d) < 1e-9:
        return dict(dc=np.nan, dead=np.nan, t10=np.nan, t50=np.nan, t90=np.nan, over=np.nan)
    frac = (resp - pre) / d
    first = lambda th: int(np.argmax(frac >= th)) + 1 if (frac >= th).any() else np.nan
    return dict(dc=d / du, dead=first(0.05), t10=first(0.10), t50=first(0.50), t90=first(0.90),
                over=(frac.max() - 1.0) * 100.0)


GAIN_FIT = np.load(Path(__file__).resolve().parent / 'gain_fit.npy')


def equilibrium_u(c0, roll, v):
    """Steer that holds lataccel at c0: steady state satisfies c_ss = roll + G(v)*u."""
    G = float(np.clip(np.polyval(GAIN_FIT, v), 0.3, 4.0))
    return (c0 - roll) / G


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'sysid'
    plant = Plant(device=DEV)
    NOM = dict(v=22.0, roll=0.0, a=0.0, c0=0.0, u0=0.0)
    DU = 0.30
    print(f'[{tag}] NREP={NREP} SETTLE={SETTLE} RESP={RESP} nominal={NOM} du={DU}', flush=True)
    hdr = f'  {"condition":26} {"DC gain":>8} {"dead":>5} {"t10":>4} {"t50":>4} {"t90":>4} {"over%":>7}'

    def sweep(name, key, values, du=DU):
        print(f'\n=== {name} ===', flush=True); print(hdr, flush=True)
        for val in values:
            c = dict(NOM); c[key] = val
            # Put the operating point AT EQUILIBRIUM. Setting c0 and u0 independently leaves the
            # plant mid-transient, and the step response then rides on an ongoing drift -- that
            # produced DC gains of 0.40-2.93 and 253% "overshoot", all artifact.
            u0 = equilibrium_u(c['c0'], c['roll'], c['v'])
            # NULL GATE: rerun the identical condition with du=0. Whatever moves is drift, not
            # dynamics. If it is a significant fraction of the step response, the measurement is
            # not trustworthy and is flagged rather than reported as a plant property.
            pre0, null = step_test(plant, c['v'], c['roll'], c['a'], c['c0'], u0, 0.0)
            drift = null[-8:].mean() - pre0
            pre, resp = step_test(plant, c['v'], c['roll'], c['a'], c['c0'], u0, du)
            m = metrics(pre, resp, du)
            flag = '  DRIFT' if abs(drift) > 0.15 * abs(m['dc'] * du) else ''
            print(f'  {key}={val:<21} {m["dc"]:8.3f} {m["dead"]:5} {m["t10"]:4} {m["t50"]:4} '
                  f'{m["t90"]:4} {m["over"]:7.1f}  u0={u0:+.3f} drift={drift:+.4f}{flag}', flush=True)

    sweep('SPEED', 'v', [5.0, 12.0, 20.0, 27.0, 34.0])
    sweep('ROAD ROLL (lataccel units)', 'roll', [-1.5, -0.5, 0.0, 0.5, 1.5])
    sweep('LONGITUDINAL ACCEL', 'a', [-2.0, -1.0, 0.0, 1.0, 2.0])
    sweep('OPERATING LATACCEL (curvature)', 'c0', [-2.0, -1.0, 0.0, 1.0, 2.0])
    # the u0 sweep is dropped: with the operating point pinned to equilibrium, u0 is determined by
    # (c0, roll, v) and is no longer a free variable. It is the c0 sweep in different units.

    print('\n=== LINEARITY: is the response proportional to step size? ===', flush=True)
    print(hdr, flush=True)
    for du in [-0.60, -0.30, -0.10, 0.10, 0.30, 0.60]:
        u0 = equilibrium_u(NOM['c0'], NOM['roll'], NOM['v'])
        pre, resp = step_test(plant, NOM['v'], NOM['roll'], NOM['a'], NOM['c0'], u0, du)
        m = metrics(pre, resp, du)
        print(f'  du={du:<+21.2f} {m["dc"]:8.3f} {m["dead"]:5} {m["t10"]:4} {m["t50"]:4} '
              f'{m["t90"]:4} {m["over"]:7.1f}', flush=True)
    print('\n  A constant DC gain column => superposition holds and every linear design in this', flush=True)
    print('  project (inverse-plant feedforward, DMC, Tikhonov reference) is well founded.', flush=True)


if __name__ == '__main__':
    main()
