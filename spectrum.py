"""Where in the FREQUENCY SPECTRUM does the cost live, and where do we lose?

The benchmark cost is exactly a frequency-weighted objective, so this decomposition is exact rather
than an analogy:

    lataccel term   5000 * mean((c - tau)^2)              flat in frequency
    jerk term       100  * mean((dc/DEL_T)^2)             a first difference, i.e. a HIGH-PASS
                                                          filter with gain |1 - e^{-i w}|^2 / DEL_T^2

By Parseval both are sums of per-frequency power, so total cost can be attributed to frequency bands
with no approximation. That makes "minimal discord" a precise question: which bands carry our error,
and which bands does the cost punish.

Three trajectories on the same segments and the same noise draws:
  * `cnn_v4`               the deliverable
  * `ff_pi_tau`            the best classical controller
  * seed-aware optimum     the best plan `steer_opt` found (an exploit; the achievable reference)

What to look for. If our excess error is concentrated in a narrow band, that is a loop-shaping defect
and a filter can fix it. If it is spread flat across the spectrum, it is broadband noise and no
filter helps -- which would be consistent with everything else this project has measured.

Usage: python spectrum.py [nseg]
"""
import sys, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, DEL_T
from oracle_build import Stepper
from steer_lookup import run_plan, score

DEV = 'cuda'


def trajectories(plant, segs, net, opt_plan):
    """cnn_v4 closed loop and the optimal plan replayed, in the same rollout machinery."""
    B, T = len(segs), COST_END_IDX
    with torch.no_grad():
        torch.manual_seed(0)
        S = Stepper(plant, segs, T)
        pol = AblPolicy(net, B, DEV)
        plan = torch.zeros(B, T, device=DEV)
        while S.t < T:
            plan[:, S.t] = pol(S.ctx()); S.step(plan[:, S.t])
    c_cnn, tg, _, _ = run_plan(plant, segs, plan.detach(), T)
    c_opt, _, _, _ = run_plan(plant, segs, opt_plan, T)
    return c_cnn, c_opt, tg


def spec(x):
    """One-sided power spectrum of each row, normalised so the sum equals mean(x^2)."""
    X = np.fft.rfft(x, axis=1)
    P = (np.abs(X) ** 2) / (x.shape[1] ** 2)
    P[:, 1:-1] *= 2.0
    return P.mean(0)


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    Z = np.load('so4_5000_128.npz', allow_pickle=True)
    files = [str(f) for f in Z['files']][:nseg]
    segs = [load_segment(f) for f in files]
    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load('cnn_v4.pt', map_location=DEV)); net.eval()
    opt_plan = torch.tensor(Z['plan'][:nseg], device=DEV)

    c_cnn, c_opt, tg = trajectories(plant, segs, net, opt_plan)
    c_cnn = c_cnn.cpu().numpy(); c_opt = c_opt.cpu().numpy(); tg = tg.cpu().numpy()
    N = c_cnn.shape[1]
    f = np.fft.rfftfreq(N, d=DEL_T)                      # Hz, Nyquist = 5 Hz

    P_tau = spec(tg)
    P_e_cnn = spec(c_cnn - tg)
    P_e_opt = spec(c_opt - tg)
    # Jerk from the spectrum of the ACTUAL difference signal, not from the analytic
    # |1 - e^{-i w}|^2 weight applied to the trajectory spectrum. The latter assumes a CIRCULAR
    # difference: the FFT wraps c[N-1] back to c[0] and injects a jump that is not in the cost. That
    # inflated the jerk term by 67% (36.20 vs the true 21.67) and broke the Parseval check.
    d_cnn = np.diff(c_cnn, axis=1) / DEL_T
    d_opt = np.diff(c_opt, axis=1) / DEL_T
    fd = np.fft.rfftfreq(d_cnn.shape[1], d=DEL_T)
    jrk_cnn_d, jrk_opt_d = 100.0 * spec(d_cnn), 100.0 * spec(d_opt)
    lat_cnn, lat_opt = 5000.0 * P_e_cnn, 5000.0 * P_e_opt

    def band(arr, freqs, lo, hi):
        return arr[(freqs >= lo) & (freqs < hi)].sum()
    print(f'  Parseval check: cnn total from spectrum {lat_cnn.sum() + jrk_cnn_d.sum():8.2f}', flush=True)
    print(f'                  cnn total direct        '
          f'{float(score(torch.tensor(c_cnn), torch.tensor(tg)).mean()):8.2f}\n', flush=True)

    BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 5.0)]
    print('=== cost by frequency band (128 segments) ===')
    print(f'  {"band (Hz)":12} {"tau power":>10} | {"cnn lat":>8} {"cnn jerk":>9} {"cnn tot":>8} | '
          f'{"opt lat":>8} {"opt jerk":>9} {"opt tot":>8} | {"gap":>7} {"gap%":>6}')
    tot_gap = (lat_cnn.sum() + jrk_cnn_d.sum()) - (lat_opt.sum() + jrk_opt_d.sum())
    for lo, hi in BANDS:
        cl, cj = band(lat_cnn, f, lo, hi), band(jrk_cnn_d, fd, lo, hi)
        ol, oj = band(lat_opt, f, lo, hi), band(jrk_opt_d, fd, lo, hi)
        g = (cl + cj) - (ol + oj)
        print(f'  {lo:4.2f}-{hi:<7.2f} {band(P_tau, f, lo, hi):10.4f} | {cl:8.2f} {cj:9.2f} {cl + cj:8.2f} | '
              f'{ol:8.2f} {oj:9.2f} {ol + oj:8.2f} | {g:7.2f} {100 * g / tot_gap:5.1f}%')
    print(f'\n  totals: cnn {lat_cnn.sum() + jrk_cnn_d.sum():.2f}   opt {lat_opt.sum() + jrk_opt_d.sum():.2f}'
          f'   gap {tot_gap:.2f}')
    print(f'  target power above 1 Hz: {100 * P_tau[f >= 1].sum() / P_tau.sum():.2f}% '
          f'-- the reference is almost entirely low frequency')
    hi_gap = sum(((band(lat_cnn, f, lo, hi) + band(jrk_cnn_d, fd, lo, hi))
                  - (band(lat_opt, f, lo, hi) + band(jrk_opt_d, fd, lo, hi)))
                 for lo, hi in BANDS if lo >= 1.0)
    print(f'  share of the gap above 1 Hz: {100 * hi_gap / tot_gap:.1f}%')


if __name__ == '__main__':
    main()
