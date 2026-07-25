"""Intermittent ("act or hold") wrapper around any base controller.

Every controller in this repo is imperative: it emits a fresh action each step whether or not
one is needed. Because the cost charges jerk on every lataccel change, each micro-adjustment buys
a little tracking and pays a little jerk -- and some of those trades lose. Measured on day one:
a zero controller has jerk 12.33 (irreducible plant noise) while PID has 18.71, so ~6 points of
jerk are bought purely by reacting.

Humans do not do this. Markkula et al. (2018) find ~91% of lane-keeping time at *zero* steering
rate, with discrete 0.4-0.6 s corrections (see also Todorov & Jordan's minimal intervention
principle, and Boer/Goodrich satisficing control).

This wrapper adds the missing null action: hold the previous command exactly unless the base
controller wants to move it by more than DEAD. A continuous network can never emit exactly
u_prev; this can.
"""
import os
import importlib
import numpy as np
from . import BaseController

BASE = os.environ.get('HOLD_BASE', 'cnn')
DEAD = float(os.environ.get('HOLD_DEAD', 0.02))     # action-change deadband
DECAY = float(os.environ.get('HOLD_DECAY', 0.0))    # optional: shrink held error over time


class Controller(BaseController):
    def __init__(self, base=None, dead=None):
        name = base or BASE
        self.inner = importlib.import_module(f'controllers.{name}').Controller()
        self.dead = DEAD if dead is None else dead
        self.u = None
        self.n_hold = 0
        self.n_step = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        want = self.inner.update(target_lataccel, current_lataccel, state, future_plan)
        self.n_step += 1
        if self.u is None:
            self.u = want
            return self.u
        if abs(want - self.u) <= self.dead:
            self.n_hold += 1                 # null action: hold the previous command exactly
            return self.u
        self.u = want
        return self.u
