"""Segment-routed portfolio: run a different strategy per segment from a lookup table.

This is a DIAGNOSTIC, not a controller. The route table is keyed on segment identity, so when it is
built from the same segments it is evaluated on, it is an oracle -- the same category of thing as the
sub-30 leaderboard exploits, and not submittable. Its purpose is to upper-bound what any causal
router could buy, and to test whether the strategies are genuinely complementary or merely
differently-noisy.

Routing key: the segment file path, passed as `seg=` (our harnesses construct the controller per
file) or via the SEG environment variable. Unrouted segments fall back to `default`.

Read `oracle_gap` alongside `null_gap` from the matched trivial-perturbation control before believing
any number this produces -- see FINDINGS_PORTFOLIO.md.
"""
import os, json, importlib


def _build(spec):
    mod, kw = spec
    return importlib.import_module('controllers.' + mod).Controller(**kw)


class Controller:
    def __init__(self, routes=None, arms=None, default=None, seg=None):
        """routes: {segment_path: arm_name}; arms: {arm_name: (module, kwargs)}."""
        if isinstance(routes, str):
            routes = json.load(open(routes))
        self.routes, self.arms = routes or {}, arms or {}
        key = seg if seg is not None else os.environ.get('SEG', '')
        name = self.routes.get(key, self.routes.get(os.path.basename(key), default))
        self.arm_name = name
        self.inner = _build(self.arms[name])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.inner.update(target_lataccel, current_lataccel, state, future_plan)
