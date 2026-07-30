"""CORRECTED gradient probe.

BUG in the original grad_probe.py: the saturation mask was computed from a rollout in `sample` mode
with the cnn policy, while the gradient came from a SEPARATE rollout in `gumbel` mode with
soft_tokens. Those consume RNG differently, so they follow different trajectories and saturate at
different steps -- the mask did not correspond to the rollout being differentiated. The reported
"6.8x larger gradient at saturated steps" was therefore comparing against the wrong steps.

Fixed: take the mask from the SAME rollout that produces the gradient.
"""
import torch, numpy as np
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, MAX_ACC_DELTA
P=Plant(device='cuda'); net=AblNet('PM').cuda()
net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
for q in net.parameters(): q.requires_grad_(False)
segs=[load_segment(p) for p in sorted(Path('data/SYNTHETIC').iterdir())[5000:5024]]
B,T=len(segs),COST_END_IDX
torch.manual_seed(0)
with torch.no_grad():
    _,_,acts = rollout(P,segs,AblPolicy(net,B,'cuda'),mode='sample',stop=T,return_actions=True)

class Free:
    def __init__(self,u): self.u=u
    def __call__(self,ctx): return self.u[:,ctx['step']]

print(f'n={B} segments, mask taken from the SAME rollout that is differentiated\n')
for st in (False,True):
    u=acts.detach().clone().requires_grad_(True)
    torch.manual_seed(0)
    traj,tgt=rollout(P,segs,Free(u),mode='gumbel',stop=T,soft_tokens=True,st_clamp=st)
    lat=traj[:,CONTROL_START_IDX:T]
    sat=((lat[:,1:]-lat[:,:-1]).abs()>=0.99*MAX_ACC_DELTA).detach()   # from THIS rollout
    cost(traj,tgt)[2].sum().backward()
    g=u.grad[:,CONTROL_START_IDX:T-1].abs()
    if sat.sum()==0:
        print(f'  st_clamp={str(st):5}  no saturated steps in this rollout'); continue
    gs=g[sat].mean().item(); gn=g[~sat].mean().item()
    print(f'  st_clamp={str(st):5}  saturated steps={int(sat.sum()):4d}/{sat.numel()}  '
          f'|grad| SAT={gs:9.3e}  normal={gn:9.3e}  ratio={gs/gn:6.3f}')
