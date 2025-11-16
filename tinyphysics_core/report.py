from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from .config import CONTROL_START_IDX, HISTOGRAM_BINS

DEFAULT_COLORS = {
  'test': '#c0392b',
  'baseline': '#2980b9'
}


def _img2base64(fig: plt.Figure) -> str:
  buf = BytesIO()
  fig.savefig(buf, format='png')
  data = b64encode(buf.getbuffer())
  return data.decode("ascii")


def plot_cost_distributions(costs_df: pd.DataFrame, colors: Dict[str, str]) -> str:
  fig, axs = plt.subplots(ncols=3, figsize=(18, 6), sharey=True)
  for ax, cost in zip(axs, ['lataccel_cost', 'jerk_cost', 'total_cost']):
    for controller in costs_df['controller'].unique():
      color = colors.get(controller, None)
      ax.hist(costs_df[costs_df['controller'] == controller][cost], bins=HISTOGRAM_BINS, label=controller, alpha=0.5, color=color)
    ax.set_xlabel('Cost')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cost Distribution: {cost}')
    ax.legend()
  encoded = _img2base64(fig)
  plt.close(fig)
  return encoded


def plot_sample_rollouts(sample_rollouts: List[Dict], colors: Dict[str, str]) -> str:
  rows = max(1, len(sample_rollouts))
  fig, axs = plt.subplots(ncols=1, nrows=rows, figsize=(15, 3 * rows), sharex=True)
  if rows == 1:
    axs = [axs]
  for ax, rollout in zip(axs, sample_rollouts):
    ax.plot(rollout['desired_lataccel'], label='Desired Lateral Acceleration', color='#27ae60')
    ax.plot(rollout['test_controller_lataccel'], label='Test Controller Lateral Acceleration', color=colors.get('test'))
    ax.plot(rollout['baseline_controller_lataccel'], label='Baseline Controller Lateral Acceleration', color=colors.get('baseline'))
    ax.set_xlabel('Step')
    ax.set_ylabel('Lateral Acceleration')
    ax.set_title(f"Segment: {rollout['seg']}")
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
  fig.tight_layout()
  encoded = _img2base64(fig)
  plt.close(fig)
  return encoded


def build_report_html(test: str, baseline: str, sample_rollouts: List[Dict], costs: List[Dict], num_segs: int, colors: Dict[str, str] | None = None) -> str:
  colors = colors or DEFAULT_COLORS
  res = []
  res.append("""
  <html>
    <head>
    <title>Comma Controls Challenge: Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap" rel="stylesheet">
    <style type='text/css'>
      table {border-collapse: collapse; font-size: 30px; margin-top: 20px; margin-bottom: 30px;}
      th, td {border: 1px solid black; text-align: left; padding: 20px;}
      th {background-color: #f2f2f2;}
      th {background-color: #f2f2f2;}
    </style>
    </head>
    <body style="font-family: 'JetBrains Mono', monospace; margin: 20px; padding: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center">
    """)
  res.append("<h1 style='font-size: 50px; font-weight: 700; text-align: center'>Comma Controls Challenge: Report</h1>")
  res.append(f"<h3 style='font-size: 30px;'><span style='background: {colors['test']}; color: #fff; padding: 10px'>Test Controller: {test}</span> ⚔️ <span style='background: {colors['baseline']}; color: #fff; padding: 10px'>Baseline Controller: {baseline}</span></h3>")

  res.append(f"<h2 style='font-size: 30px; margin-top: 50px'>Aggregate Costs (total rollouts: {num_segs})</h2>")
  res_df = pd.DataFrame(costs)
  agg_plot = plot_cost_distributions(res_df, colors)
  res.append(f'<img style="max-width:100%" src="data:image/png;base64,{agg_plot}" alt="Plot">')
  agg_df = res_df.groupby('controller').agg({'lataccel_cost': 'mean', 'jerk_cost': 'mean', 'total_cost': 'mean'}).round(3).reset_index()
  res.append(agg_df.to_html(index=False))

  passed_baseline = agg_df[agg_df['controller'] == 'test']['total_cost'].values[0] < agg_df[agg_df['controller'] == 'baseline']['total_cost'].values[0]
  if passed_baseline:
    res.append(f"<h3 style='font-size: 20px; color: #27ae60'> ✅ Test Controller ({test}) passed Baseline Controller ({baseline})! ✅ </h3>")
    res.append("""<p>Check the leaderboard
      <a href='https://comma.ai/leaderboard'>here</a>
      and submit your results
      <a href='https://docs.google.com/forms/d/e/1FAIpQLSc_Qsh5egoseXKr8vI2TIlsskd6nNZLNVuMJBjkogZzLe79KQ/viewform'>here</a>
      !</p>""")
  else:
    res.append(f"<h3 style='font-size: 20px; color: #c0392b'> ❌ Test Controller ({test}) failed to beat Baseline Controller ({baseline})! ❌</h3>")

  res.append("<hr style='border: #ddd 1px solid; width: 80%'>")
  res.append("<h2  style='font-size: 30px; margin-top: 50px'>Sample Rollouts</h2>")
  rollout_plot = plot_sample_rollouts(sample_rollouts, colors)
  res.append(f'<img style="max-width:100%" src="data:image/png;base64,{rollout_plot}" alt="Plot">')
  res.append("</body></html>")
  return "\n".join(res)


def save_report(html: str, output_path: Path | str = "report.html") -> None:
  output_path = Path(output_path)
  with open(output_path, "w", encoding='utf-8') as fob:
    fob.write(html)
