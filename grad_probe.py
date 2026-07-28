"""Does the hard rate clamp actually zero the gradient, and does st_clamp restore it?

Actions are made free leaf tensors (initialised from cnn's own actions, so the trajectory and its
saturation pattern are realistic). Backprop the benchmark cost to those actions and compare the
gradient reaching SATURATED steps vs normal ones, with and without the straight-through clamp.
"""
import torch, numpy as np
from pathlib import Path
from torch_sim import Plant, load_segment, rollout, cost
from nets import AblNet, AblPolicy
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, MAX_ACC_DELTA

P=Plant(device='cuda'); net=AblNet('PM').cuda()
net.load_state_dict(torch.load('cnn_PM.pt')); net.eval()
for q in net.parameters(): q.requires_grad_(False)
segs=[load_segment(p) for p in sorted(Path('data/SYNTHETIC').iterdir())[5000:5032]]
B,T=len(segs),COST_END_IDX

# 1. get cnn's own actions and find where the plant's rate clamp binds
torch.manual_seed(0)
with torch.no_grad():
    tr,tg,acts = rollout(P,segs,AblPolicy(net,B,'cuda'),mode='sample',stop=T,return_actions=True)
lat=tr[:,CONTROL_START_IDX:T]
sat=(lat[:,1:]-lat[:,:-1]).abs()>=0.99*MAX_ACC_DELTA      # [B, T-101]
print(f'n={B} segments | saturated steps: {int(sat.sum())} of {sat.numel()} = {100*sat.float().mean():.2f}%')

class Free:
    def __init__(self,u): self.u=u
    def __call__(self,ctx): return self.u[:,ctx['step']]

for st in (False,True):
    u=acts.detach().clone().requires_grad_(True)
    torch.manual_seed(0)
    traj,tgt=rollout(P,segs,Free(u),mode='gumbel',stop=T,soft_tokens=True,st_clamp=st)
    cost(traj,tgt)[2].sum().backward()
    g=u.grad[:,CONTROL_START_IDX:T-1].abs()               # align with sat
    gs=g[sat].mean().item(); gn=g[~sat].mean().item()
    print(f'  st_clamp={str(st):5}  |grad| at SATURATED steps={gs:.3e}   at normal steps={gn:.3e}   ratio={gs/gn:.3f}')
