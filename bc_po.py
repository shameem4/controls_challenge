"""Policy optimisation through the differentiable plant + a behaviour-cloning auxiliary loss.

Targets the exact failure plain BC hit. The student cloned the `steer_lookup` teacher to R^2 0.974
per-step but scored 58.8 vs cnn_v2's 50.7 in closed loop, because ~5% per-step action error compounds
over 400 steps into states the teacher never visited. Pure BC only ever sees the TEACHER's state
distribution.

    loss = PO_cost(student's own rollout)  +  alpha * MSE(student(teacher_obs), u_teacher)

The PO term is computed on the student's OWN trajectory through the plant, so it is trained where it
actually goes -- that is what fixes distribution shift. The BC term pulls it toward a teacher that
genuinely beats it (37.8 vs 42.7 on the same segments). Standard demonstration-augmented RL.

Built-in matched control: alpha=0 is pure policy optimisation from the same init, same seed, same
budget. Any gain at alpha>0 is attributable to the teacher signal rather than to extra training --
which is the comparison the cnn_dual promotion failed to make earlier in this project.

Scale note: PO cost is ~45 while BC MSE is ~0.002, so alpha must be large (hundreds to thousands) for
the two gradients to be comparable. Sweep it.

Usage: python bc_po.py <iters> [tag]    env: SEED, LR, ALPHA, INIT, DATA
"""
import sys, os, glob, numpy as np, torch
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from tinyphysics import COST_END_IDX
from bc_train import load, to_inputs

DEV = 'cuda'


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    tag = sys.argv[2] if len(sys.argv) > 2 else 'bcpo'
    seed = int(os.environ.get('SEED', 0))
    lr = float(os.environ.get('LR', 1e-4))
    alpha = float(os.environ.get('ALPHA', 500.0))
    init = os.environ.get('INIT', 'cnn_v2.pt')
    data = os.environ.get('DATA', 'slA_2000_128.npz')
    bs, ACC, TBPTT = 8, 4, 60

    torch.manual_seed(seed); np.random.seed(seed)
    ALL = sorted(Path('data/SYNTHETIC').iterdir())
    TRAIN, VAL = ALL[2000:4000], ALL[4000:4200]

    obs, ui = load(sorted(glob.glob(data)))
    ffw, v, fb = to_inputs(obs, ui)
    FFW = torch.tensor(ffw, device=DEV); VV = torch.tensor(v, device=DEV)
    FB = torch.tensor(fb, device=DEV); Y = torch.tensor(ui, dtype=torch.float32, device=DEV)
    nd = FFW.shape[0]

    plant = Plant(device=DEV)
    net = AblNet('PM').to(DEV)
    net.load_state_dict(torch.load(init, map_location=DEV))
    opt = torch.optim.Adam(net.parameters(), lr)
    os.makedirs('ckpts', exist_ok=True)
    print(f'[{tag}] init={init} alpha={alpha} lr={lr} seed={seed} iters={iters} '
          f'teacher_segs={nd}', flush=True)

    def val(seeds=(0, 1)):
        tots = []
        for s in seeds:
            torch.manual_seed(s)
            with torch.no_grad():
                t = []
                for i in range(0, 64, bs):
                    segs = [load_segment(f) for f in VAL[i:i + bs]]
                    traj, target = rollout(plant, segs, AblPolicy(net, len(segs), DEV),
                                           mode='sample', stop=COST_END_IDX)
                    t.append(cost(traj, target)[2])
                tots.append(torch.cat(t).mean().item())
        return float(np.mean(tots))

    print(f'[{tag}] init val = {val():.3f}', flush=True)
    best = 1e9
    for it in range(iters):
        opt.zero_grad()
        po_tot = bc_tot = 0.0
        for _ in range(ACC):
            # --- policy optimisation on the student's OWN state distribution ---
            idx = torch.randint(len(TRAIN), (bs,))
            segs = [load_segment(TRAIN[k]) for k in idx]
            traj, target = rollout(plant, segs, AblPolicy(net, bs, DEV), mode='gumbel',
                                   tbptt=TBPTT, stop=COST_END_IDX, soft_tokens=True)
            po = cost(traj, target)[2].mean()
            # --- behaviour cloning against the teacher, on the teacher's states ---
            if alpha > 0:
                j = torch.randint(nd, (bs,), device=DEV)
                pred = net(FFW[j].reshape(-1, FFW.shape[-1]),
                           VV[j].reshape(-1),
                           FB[j].reshape(-1, FB.shape[-1]))
                bc = ((pred - Y[j].reshape(-1)) ** 2).mean()
            else:
                bc = torch.zeros((), device=DEV)
            ((po + alpha * bc) / ACC).backward()
            po_tot += po.item() / ACC; bc_tot += bc.item() / ACC
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (it + 1) % 25 == 0 or it == iters - 1:
            vl = val()
            mark = ''
            if vl < best:
                best, mark = vl, ' *'
                torch.save(net.state_dict(), f'ckpts/{tag}_best.pt')
            print(f'[{tag}] it{it + 1:4d} po={po_tot:6.2f} bc={bc_tot:.5f} val={vl:6.3f}{mark}', flush=True)
    print(f'[{tag}] done best_val={best:.3f} -> ckpts/{tag}_best.pt', flush=True)


if __name__ == '__main__':
    main()
