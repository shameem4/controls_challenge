import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def main(report_path: Path, data_dir: Path) -> None:
  st.set_page_config(page_title="Report Visualizer", layout="wide")
  st.title("Comma Controls Challenge: Report Visualizer")
  summary_df, step_df = load_report(report_path)

  if summary_df.empty or step_df.empty:
    st.error("report.csv is empty or missing expected columns. Please re-run eval.py before using the visualizer.")
    return

  controllers = sorted(summary_df['controller'].unique())
  segments = sorted(summary_df['segment'].unique())
  if 'selected_segment' not in st.session_state:
    st.session_state['selected_segment'] = segments[0]

  st.sidebar.header("Filters")
  selected_controllers = st.sidebar.multiselect("Controllers", controllers, default=controllers)
  summary_metrics = st.sidebar.multiselect(
    "Summary Metrics",
    ['total_cost', 'lataccel_cost', 'jerk_cost', 'avg_pid_term_abs', 'max_action_abs', 'integrator_clamp_steps', 'clamp_relief_events'],
    default=['total_cost', 'lataccel_cost', 'jerk_cost']
  )

  st.subheader("Segment Selection")
  cost_columns = ['total_cost', 'lataccel_cost', 'jerk_cost']
  segment_table = summary_df.pivot_table(index='segment', columns='controller', values=cost_columns)
  segment_table.columns = [f"{controller}_{metric}" for metric, controller in segment_table.columns]
  segment_table.reset_index(inplace=True)
  segment_table.insert(0, 'selected', segment_table['segment'] == st.session_state['selected_segment'])
  edited_table = st.data_editor(
    segment_table,
    column_config={
      'selected': st.column_config.CheckboxColumn('Select', help="Check to view this segment")
    },
    hide_index=True,
    use_container_width=True
  )
  edited_selected = edited_table[edited_table['selected']]
  if len(edited_selected) > 1:
    first_segment = edited_selected.iloc[0]['segment']
    st.session_state['selected_segment'] = first_segment
    edited_table.loc[:, 'selected'] = edited_table['segment'] == first_segment
  elif len(edited_selected) == 1:
    st.session_state['selected_segment'] = edited_selected.iloc[0]['segment']
  selected_segment = st.session_state['selected_segment']

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

  segment_steps = step_df[step_df['segment'] == selected_segment]
  segment_df = load_segment_csv(selected_segment, data_dir)

  st.subheader("Combined Step & Raw Metrics")
  plot_df = pd.DataFrame()
  if not segment_steps.empty:
    plot_df['step'] = sorted(segment_steps['step'].unique())
    plot_df.set_index('step', inplace=True)
    metric_labels = {
      'controller_lataccel': 'lataccel',
      'lat_error': 'lat_error',
      'action': 'action',
      'jerk': 'jerk'
    }
    for metric, label in metric_labels.items():
      pivot = segment_steps.pivot_table(index='step', columns='controller', values=metric)
      for controller in selected_controllers:
        if controller in pivot.columns:
          plot_df[f"{controller} {label}"] = pivot[controller]
    target_series = segment_steps[['step', 'desired_lataccel']].drop_duplicates('step').set_index('step')['desired_lataccel']
    plot_df['Desired target_lataccel'] = target_series
  raw_velocity_label = None
  if not segment_df.empty:
    raw_df = segment_df.copy()
    raw_df['step'] = raw_df.index
    raw_df = raw_df.set_index('step')
    raw_columns = {
      'targetLateralAcceleration': 'Raw target_lataccel',
      'steerCommand': 'Raw steer_command',
      'vEgo': 'Raw v_ego',
      'aEgo': 'Raw a_ego',
      'roll': 'Raw roll'
    }
    for col, label in raw_columns.items():
      if col in raw_df.columns:
        plot_df[label] = raw_df[col]
        if col == 'vEgo':
          raw_velocity_label = label
  if plot_df.empty:
    st.info("No data available for combined plot.")
  else:
    available_series = [col for col in plot_df.columns if col != 'step']
    default_series = [series for series in available_series if 'lataccel' in series][:4]
    selected_series = []
    color_map: Dict[str, str] = {}
    fallback_colors = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
      "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    color_cycle = iter(fallback_colors)
    with st.expander("Plot Series Selection", expanded=True):
      for series in available_series:
        if series not in color_map:
          color_map[series] = next(color_cycle, "#1f77b4")
        color_box = f"<span style='display:inline-block;width:12px;height:12px;background:{color_map[series]};border-radius:2px;'></span>"
        check_col, label_col = st.columns([0.1, 0.9])
        checked = check_col.checkbox(" ", value=series in default_series, key=f"plot_{series}_{selected_segment}", label_visibility='hidden')
        label_col.markdown(f"<span style='display:flex;align-items:center;gap:8px'>{color_box}<span>{series}</span></span>", unsafe_allow_html=True)
        if checked:
          selected_series.append(series)
    if not selected_series:
      st.info("Select at least one series to visualize.")
    else:
      fig = make_subplots(specs=[[{"secondary_y": True}]])
      for series in selected_series:
        use_secondary = bool(raw_velocity_label and series == raw_velocity_label)
        fig.add_trace(
          go.Scatter(
            x=plot_df.index,
            y=plot_df[series],
            name=series,
            line=dict(color=color_map.get(series, None))
          ),
          secondary_y=use_secondary
        )
      def _symmetric_range(values: pd.Series) -> Tuple[float, float]:
        max_abs = max(values.abs().max(), 1e-6)
        padding = max_abs * 0.05
        return (-max_abs - padding, max_abs + padding)

      primary_values = pd.Series(dtype=float)
      for series in selected_series:
        if series != raw_velocity_label:
          primary_values = pd.concat([primary_values, plot_df[series].dropna()])
      if not primary_values.empty:
        fig.update_yaxes(title_text="Primary Metrics", secondary_y=False, range=_symmetric_range(primary_values))
      else:
        fig.update_yaxes(title_text="Primary Metrics", secondary_y=False)
      if raw_velocity_label and raw_velocity_label in selected_series:
        velocity_values = plot_df[raw_velocity_label].dropna()
        fig.update_yaxes(title_text="Velocity", secondary_y=True, range=_symmetric_range(velocity_values))
      else:
        fig.update_yaxes(title_text="Velocity", secondary_y=True)
      fig.update_layout(legend=dict(orientation='h'))
      st.plotly_chart(fig, use_container_width=True)

  st.subheader("Detailed Step Table")
  max_rows = st.slider("Rows to display", min_value=100, max_value=2000, value=500, step=100)
  st.dataframe(
    segment_steps[segment_steps['controller'].isin(selected_controllers)]
    .sort_values(['controller', 'step'])
    .head(max_rows)
  )

  if segment_df.empty:
    st.info(f"No CSV found for segment {selected_segment}")

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
