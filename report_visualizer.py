import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors as plotly_colors
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from tinyphysics_core.config import CONTROL_START_IDX, DATASET_PATH, DEL_T

COLOR_PALETTE = list(dict.fromkeys(
  plotly_colors.qualitative.Alphabet +
  plotly_colors.qualitative.Bold +
  plotly_colors.qualitative.Dark24 +
  plotly_colors.qualitative.Light24 +
  plotly_colors.qualitative.Plotly
))


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
  if not segments:
    st.warning("No segments available in report.csv")
    return

  st.sidebar.header("Filters")
  selected_controllers = st.sidebar.multiselect("Controllers", controllers, default=controllers)
  if not selected_controllers:
    st.warning("Select at least one controller to visualize results.")
    return
  summary_metrics = st.sidebar.multiselect(
    "Summary Metrics",
    ['total_cost', 'lataccel_cost', 'jerk_cost', 'avg_pid_term_abs', 'max_action_abs', 'integrator_clamp_steps', 'clamp_relief_events'],
    default=['total_cost', 'lataccel_cost', 'jerk_cost']
  )

  st.subheader("Segment Selection")
  cost_columns = ['total_cost']
  segment_table = summary_df.pivot_table(index='segment', columns='controller', values=cost_columns)
  segment_table.columns = [f"{controller}_{metric}" for metric, controller in segment_table.columns]
  segment_table.reset_index(inplace=True)
  builder = GridOptionsBuilder.from_dataframe(segment_table)
  builder.configure_selection('single', use_checkbox=False)
  builder.configure_default_column(sortable=True, filter=True)
  grid_options = builder.build()
  grid_response = AgGrid(
    segment_table,
    gridOptions=grid_options,
    height=120,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    theme='streamlit',
    fit_columns_on_grid_load=True
  )
  selected_rows = grid_response.get('selected_rows')
  if isinstance(selected_rows, pd.DataFrame):
    selected_rows = selected_rows.to_dict('records')
  elif selected_rows is None:
    selected_rows = []
  if selected_rows:
    selected_segment = selected_rows[0].get('segment', segments[0])
  else:
    selected_segment = segments[0]



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
    if 'series_selection' not in st.session_state:
      st.session_state['series_selection'] = {series: (series in default_series) for series in available_series}
    else:
      for series in available_series:
        st.session_state['series_selection'].setdefault(series, series in default_series)
    if 'series_colors' not in st.session_state:
      st.session_state['series_colors'] = {}
    selected_series = []
    color_map: Dict[str, str] = {}
    with st.expander("Plot Series Selection", expanded=True):
      selection_area = st.container(height=220)
      column_wrappers = selection_area.columns(3)
      for idx, series in enumerate(available_series):
        if series not in st.session_state['series_colors']:
          used_colors = set(st.session_state['series_colors'].values())
          available_color = next((c for c in COLOR_PALETTE if c not in used_colors), COLOR_PALETTE[len(used_colors) % len(COLOR_PALETTE)])
          st.session_state['series_colors'][series] = available_color
        color_map[series] = st.session_state['series_colors'][series]
        color_box = f"<span style='display:inline-block;width:12px;height:12px;background:{color_map[series]};border-radius:2px;'></span>"
        col_wrapper = column_wrappers[idx % 3]
        check_col, label_col = col_wrapper.columns([0.25, 0.75])
        checkbox_key = f"plot_series_{series}"
        checked = check_col.checkbox(" ", value=st.session_state['series_selection'][series], key=checkbox_key, label_visibility='hidden')
        st.session_state['series_selection'][series] = checked
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
      fig.update_layout(legend=dict(orientation='h'), title=f"Segment {selected_segment}")
      st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'responsive': True})

  st.subheader("Segment Summary")
  display_rows = summary_df[
    (summary_df['segment'] == selected_segment) &
    (summary_df['controller'].isin(selected_controllers))
  ]
  if display_rows.empty:
    st.info("No summary rows for the current selection.")
  else:
    columns = ['controller'] + summary_metrics
    st.dataframe(display_rows[columns].set_index('controller'), width='stretch')


  st.subheader("Detailed Step Table")
  max_rows = st.slider("Rows to display", min_value=100, max_value=2000, value=500, step=100)
  st.dataframe(
    segment_steps[segment_steps['controller'].isin(selected_controllers)]
    .sort_values(['controller', 'step'])
    .head(max_rows)
  , width='stretch')

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
