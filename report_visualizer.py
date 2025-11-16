import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

from tinyphysics_core.config import CONTROL_START_IDX, DATASET_PATH, DEL_T


@st.cache_data(show_spinner=False)
def load_report(report_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
  df = pd.read_csv(report_path)
  df['segment'] = df['segment'].astype(str).str.zfill(5)
  summary = df[df['row_type'] == 'summary'].copy()
  step = df[df['row_type'] == 'step'].copy()
  return summary, step


@st.cache_data(show_spinner=False)
def load_segment_csv(segment_id: str, data_dir: Path) -> pd.DataFrame:
  seg_path = data_dir / f"{int(segment_id):05d}.csv"
  if not seg_path.exists():
    return pd.DataFrame()
  segment_df = pd.read_csv(seg_path)
  return segment_df


def _segment_step_plot(step_df: pd.DataFrame, selected_controllers: List[str], metric: str) -> pd.DataFrame:
  filtered = step_df[step_df['controller'].isin(selected_controllers)]
  if filtered.empty:
    return pd.DataFrame()
  pivot = filtered.pivot_table(index='step', columns='controller', values=metric)
  if metric != 'desired_lataccel':
    target_series = filtered[['step', 'desired_lataccel']].drop_duplicates('step').set_index('step')
    pivot = pivot.join(target_series.rename(columns={'desired_lataccel': 'desired_lataccel'}), how='left')
  pivot.sort_index(inplace=True)
  return pivot


def main(report_path: Path, data_dir: Path) -> None:
  st.set_page_config(page_title="Report Visualizer", layout="wide")
  st.title("Comma Controls Challenge: Report Visualizer")
  summary_df, step_df = load_report(report_path)

  if summary_df.empty or step_df.empty:
    st.error("report.csv is empty or missing expected columns. Please re-run eval.py before using the visualizer.")
    return

  controllers = sorted(summary_df['controller'].unique())
  segments = sorted(summary_df['segment'].unique())

  st.sidebar.header("Filters")
  selected_controllers = st.sidebar.multiselect("Controllers", controllers, default=controllers)
  selected_segment = st.sidebar.selectbox("Segment", segments)
  metric_options = ['controller_lataccel', 'desired_lataccel', 'lat_error', 'action', 'jerk']
  selected_metric = st.sidebar.selectbox("Step Metric", metric_options, index=0)
  summary_metrics = st.sidebar.multiselect(
    "Summary Metrics",
    ['total_cost', 'lataccel_cost', 'jerk_cost', 'avg_pid_term_abs', 'max_action_abs', 'integrator_clamp_steps', 'clamp_relief_events'],
    default=['total_cost', 'lataccel_cost', 'jerk_cost']
  )

  st.subheader("Segment Summary")
  display_rows = summary_df[
    (summary_df['segment'] == selected_segment) &
    (summary_df['controller'].isin(selected_controllers))
  ]
  if display_rows.empty:
    st.info("No summary rows for the current selection.")
  else:
    columns = ['controller'] + summary_metrics
    st.dataframe(display_rows[columns].set_index('controller'))

  st.subheader("Step-Level Metrics")
  segment_steps = step_df[step_df['segment'] == selected_segment]
  step_plot_df = _segment_step_plot(segment_steps, selected_controllers, selected_metric)
  if step_plot_df.empty:
    st.info("No step data available for the selection.")
  else:
    st.line_chart(step_plot_df)

  st.subheader("Detailed Step Table")
  max_rows = st.slider("Rows to display", min_value=100, max_value=2000, value=500, step=100)
  st.dataframe(
    segment_steps[segment_steps['controller'].isin(selected_controllers)]
    .sort_values(['controller', 'step'])
    .head(max_rows)
  )

  st.subheader("Raw Segment Data")
  segment_df = load_segment_csv(selected_segment, data_dir)
  if segment_df.empty:
    st.info(f"No CSV found for segment {selected_segment}")
  else:
    cols_to_show = ['t', 'targetLateralAcceleration', 'steerCommand', 'vEgo', 'aEgo', 'roll']
    cols_to_show = [col for col in cols_to_show if col in segment_df.columns]
    st.line_chart(segment_df[cols_to_show].set_index('t'))
    st.caption("Raw values loaded from the data directory.")

  st.sidebar.markdown("---")
  st.sidebar.write(f"report: `{report_path}`")
  st.sidebar.write(f"data dir: `{data_dir}`")
  st.sidebar.caption(f"Showing steps from CONTROL_START_IDX={CONTROL_START_IDX}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Interactive viewer for report.csv outputs.")
  parser.add_argument("--report_path", type=Path, default=Path("report.csv"))
  parser.add_argument("--data_dir", type=Path, default=DATASET_PATH)
  args = parser.parse_args()
  main(args.report_path, Path(args.data_dir))
