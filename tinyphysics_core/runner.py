from typing import List, Tuple

import matplotlib.pyplot as plt

from .config import CONTROL_START_IDX, CONTEXT_LENGTH
from .simulator import TinyPhysicsSimulator
from .types import RolloutResult


class RolloutRunner:
  """Encapsulates the bookkeeping required to run and visualize a rollout."""

  def __init__(self, simulator: TinyPhysicsSimulator, debug: bool = False) -> None:
    self.simulator = simulator
    self.debug = debug
    self._axes: List[plt.Axes] = []

  def _init_debug_plot(self) -> None:
    if not self.debug:
      return
    plt.ion()
    fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)
    self._axes = list(ax)

  def _plot_data(self, ax, lines: List[Tuple[List[float], str]], axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
    ax.set_title(f"{title} | Step: {self.simulator.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def _render_debug(self) -> None:
    if not self.debug or not self._axes:
      return
    sim = self.simulator
    print(f"Step {sim.step_idx:<5}: Current lataccel: {sim.current_lataccel:>6.2f}, Target lataccel: {sim.target_lataccel_history[-1]:>6.2f}")
    self._plot_data(self._axes[0], [(sim.target_lataccel_history, 'Target lataccel'), (sim.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
    self._plot_data(self._axes[1], [(sim.action_history, 'Action')], ['Step', 'Action'], 'Action')
    roll_lataccel = [state.roll_lataccel for state in sim.state_history]
    v_ego = [state.v_ego for state in sim.state_history]
    self._plot_data(self._axes[2], [(roll_lataccel, 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
    self._plot_data(self._axes[3], [(v_ego, 'v_ego')], ['Step', 'v_ego'], 'v_ego')
    plt.pause(0.01)

  def run(self) -> RolloutResult:
    self.simulator.reset()
    self._init_debug_plot()
    for _ in range(CONTEXT_LENGTH, self.simulator.total_steps):
      self.simulator.step()
      if self.debug and self.simulator.step_idx % 10 == 0:
        self._render_debug()

    if self.debug:
      plt.ioff()
      plt.show()
    return self.simulator.build_rollout_result()
