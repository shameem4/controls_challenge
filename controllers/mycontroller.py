from collections import namedtuple
from controllers import BaseController

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


from collections import deque
import numpy as np
import control
import itertools



import control
import numpy as np
from scipy import signal



class Controller(BaseController):
    def __init__(self):
        super().__init__()
        self.p = 0.3
        self.i = 0.05
        self.d = 0.1

        self.dt = 0.01  # Sampling time
        self.N = 100.0  # Derivative filter coefficient

        # Build PID transfer function manually
        P = signal.TransferFunction([self.p], [1])
        I = signal.TransferFunction([self.i], [1, 0])
        D = signal.TransferFunction([self.d * self.N, 0], [1, self.N])

        # Add transfer functions (common denominator)
        num_P, den_P = P.num, P.den
        num_I, den_I = I.num, I.den
        num_D, den_D = D.num, D.den

        den_common = np.polymul(np.polymul(den_P, den_I), den_D)
        num_P = np.polymul(num_P, np.polymul(den_I, den_D))
        num_I = np.polymul(num_I, np.polymul(den_P, den_D))
        num_D = np.polymul(num_D, np.polymul(den_P, den_I))
        num_common = np.polyadd(np.polyadd(num_P, num_I), num_D)

        self.sys_c = signal.TransferFunction(num_common, den_common)

        # Discretize
        self.sys_d = self.sys_c.to_discrete(self.dt, method='bilinear')

        # State-space form
        self.A, B, C, D = signal.tf2ss(self.sys_d.num, self.sys_d.den)

        # Flatten once
        self.B = B.flatten()   # (n, )
        self.C = C.flatten()   # (n, )
        self.D = D.item()      # scalar

        # Initialize states
        self.x = np.zeros(self.A.shape[0])

    def update(self, target_lataccel, current_lataccel, state, future_plan, step_idx, action):
        error = target_lataccel - current_lataccel

        # Optimized math: no flatten() calls at runtime
        self.x = self.A @ self.x + self.B * error
        y = self.C @ self.x + self.D * error

        return y

