"""Identify plant gain G(v) from real logged data (in-distribution).
Plant model: lataccel ~= G(v)*steer_sim + roll_lataccel, steer_sim = -steerCommand.
Regress (target_lataccel - roll_lataccel) on steer_sim, binned by v_ego.
"""
import numpy as np, pandas as pd
from pathlib import Path

ACC_G = 9.81
files = sorted(Path('data/SYNTHETIC').iterdir())[:500]
X, Y, V = [], [], []
for f in files:
    df = pd.read_csv(f)
    roll_la = np.sin(df['roll'].values) * ACC_G
    steer = -df['steerCommand'].values
    y = df['targetLateralAcceleration'].values - roll_la
    v = df['vEgo'].values
    X.append(steer); Y.append(y); V.append(v)
X = np.concatenate(X); Y = np.concatenate(Y); V = np.concatenate(V)

edges = np.arange(0, 45, 5)
print("v_bin   n      G(v)=slope   R^2")
vv, gg = [], []
for lo in edges:
    m = (V >= lo) & (V < lo + 5) & (np.abs(X) > 0.02)
    if m.sum() < 500:
        continue
    x, y = X[m], Y[m]
    g = np.sum(x * y) / np.sum(x * x)          # slope through origin
    r2 = 1 - np.sum((y - g * x) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"{lo:>2}-{lo+5:<3} {m.sum():>7} {g:>10.4f} {r2:>8.3f}")
    vv.append(lo + 2.5); gg.append(g)

vv, gg = np.array(vv), np.array(gg)
for deg in (1, 2):
    c = np.polyfit(vv, gg, deg)
    print(f"G(v) deg{deg}: {np.round(c,6)}  maxresid={np.max(np.abs(np.polyval(c,vv)-gg)):.4f}")
np.save('gain_fit.npy', np.polyfit(vv, gg, 2))
print("saved gain_fit.npy")
