"""
xr_groups.py
=============
Group / F-formation-aware analyses that sit on top of xr_loaders + xr_analysis:

1. ConfLab -- 17 hand-labeled F-formation "episodes" (each: a fixed set of
   participants + a fixed time window), possibly several running concurrently,
   against the full multi-participant session. Classifies every
   simultaneously-tracked pair at every moment as "within" (same active
   labeled group) or "cross" (not in the same active group -- includes
   pairs where one or both participants have no active group label).

2. Hybrid -- pre-segmented per-group/condition/task files (already exactly
   the dyad/triad for that task), pooled by condition x task across groups.

Also provides a distribution-style proxemics plot (normalized histogram +
KDE + Hall-zone boundary lines) matching the reference figures.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from xr_loaders import load_session

HALL_BOUNDARIES = [(0.46, "Intimate\nDistance"), (1.2, "Personal\nDistance"), (3.6, "Social\nDistance")]


# ---------------------------------------------------------------------------
# ConfLab: episode parsing + loading
# ---------------------------------------------------------------------------
_EPISODE_RE = re.compile(r"conflab_(ep\d+)_([\d_]+)\.pkl$")


def parse_conflab_episode_filename(path: Path) -> tuple[str, list[str]]:
    m = _EPISODE_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Filename doesn't match conflab_epNN_p1_p2_...pkl pattern: {path}")
    episode_id, participants_str = m.groups()
    return episode_id, participants_str.split("_")


def load_conflab_episode_variant(dir_path: Path, variant_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every episode pkl in `dir_path` (conflab_head/ or conflab_shoulders/).
    Returns (episodes_long_df, intervals_df):
      episodes_long_df -- unified schema + `episode_id` column, all episodes concatenated
      intervals_df -- one row per episode: episode_id, participants (list), t0, t1, n_people
    """
    dir_path = Path(dir_path)
    frames, interval_rows = [], []
    for p in sorted(dir_path.glob("conflab_ep*.pkl")):
        episode_id, participants_from_name = parse_conflab_episode_filename(p)
        df = load_session(p, session_id=f"{variant_label}_{episode_id}", format="pkl_long")
        df["episode_id"] = episode_id
        frames.append(df)
        participants = sorted(df["participant_id"].unique(), key=lambda x: int(x))
        interval_rows.append({
            "episode_id": episode_id,
            "participants": participants,
            "t0": df["t_s"].min(),
            "t1": df["t_s"].max(),
            "n_people": len(participants),
        })
    episodes_long_df = pd.concat(frames, ignore_index=True)
    intervals_df = pd.DataFrame(interval_rows)
    return episodes_long_df, intervals_df


def load_conflab_full_session(csv_path: Path) -> pd.DataFrame:
    """Lightweight loader for the full (all ~45 participants) raw ConfLab
    session CSV -- only pulls the columns proxemics needs (position), since
    this file is large and we don't need rotation/hands for this analysis."""
    raw = pd.read_csv(csv_path, usecols=["timestamp", "uuid", "position_x", "position_y", "position_z"])
    raw = raw.dropna(subset=["uuid", "timestamp"])
    out = pd.DataFrame({
        "participant_id": raw["uuid"].astype(str),
        "t_s": pd.to_datetime(raw["timestamp"], utc=True, format="mixed").astype("int64") / 1e9,
        "position_x": raw["position_x"],
        "position_y": raw["position_y"],
        "position_z": raw["position_z"],
    })
    return out.sort_values(["participant_id", "t_s"]).reset_index(drop=True)


def build_position_grid(full_df: pd.DataFrame, rate_hz: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Resample every participant in `full_df` onto one shared uniform grid.
    Returns (t_grid, participant_ids, position_array) where position_array
    has shape (n_times, n_participants, 3); NaN outside a participant's own
    tracked range (no extrapolation)."""
    participant_ids = sorted(full_df["participant_id"].unique(), key=lambda x: int(x))
    t0, t1 = full_df["t_s"].min(), full_df["t_s"].max()
    t_grid = np.arange(t0, t1, 1.0 / rate_hz)
    pos = np.full((len(t_grid), len(participant_ids), 3), np.nan)
    for j, pid in enumerate(participant_ids):
        d = full_df[full_df["participant_id"] == pid].sort_values("t_s").drop_duplicates("t_s")
        if len(d) < 2:
            continue
        pt = d["t_s"].to_numpy()
        for k, c in enumerate(["position_x", "position_y", "position_z"]):
            # only interpolate within this participant's own tracked span -- no extrapolation
            interp = np.interp(t_grid, pt, d[c].to_numpy(), left=np.nan, right=np.nan)
            interp[(t_grid < pt.min()) | (t_grid > pt.max())] = np.nan
            pos[:, j, k] = interp
    return t_grid, participant_ids, pos


def build_group_grid(intervals_df: pd.DataFrame, t_grid: np.ndarray, participant_ids: list[str]) -> np.ndarray:
    """(n_times, n_participants) array of the active episode_id (or None) for
    each participant at each timestamp, from the interval table."""
    idx_of = {pid: j for j, pid in enumerate(participant_ids)}
    grid = np.full((len(t_grid), len(participant_ids)), None, dtype=object)
    for _, row in intervals_df.iterrows():
        t_mask = (t_grid >= row["t0"]) & (t_grid <= row["t1"])
        for pid in row["participants"]:
            j = idx_of.get(pid)
            if j is not None:
                grid[t_mask, j] = row["episode_id"]
    return grid


def classify_formation_distances(intervals_df: pd.DataFrame, full_df: pd.DataFrame,
                                  rate_hz: float = 1.0) -> dict[str, np.ndarray]:
    """Pools pairwise 3D distances across the whole session into 'within'
    (same currently-active labeled F-formation) and 'cross' (not -- covers
    different-group and ungrouped pairs) buckets."""
    t_grid, participant_ids, pos = build_position_grid(full_df, rate_hz)
    group_grid = build_group_grid(intervals_df, t_grid, participant_ids)

    n_p = len(participant_ids)
    iu, ju = np.triu_indices(n_p, k=1)
    within, cross = [], []
    for t in range(len(t_grid)):
        p = pos[t]
        valid = np.isfinite(p).all(axis=1)
        both_valid = valid[iu] & valid[ju]
        if not both_valid.any():
            continue
        diff = p[iu[both_valid]] - p[ju[both_valid]]
        dist = np.linalg.norm(diff, axis=1)
        g = group_grid[t]
        gi, gj = g[iu[both_valid]], g[ju[both_valid]]
        same = np.array([(a is not None) and (a == b) for a, b in zip(gi, gj)])
        within.append(dist[same])
        cross.append(dist[~same])
    return {
        "within": np.concatenate(within) if within else np.array([]),
        "cross": np.concatenate(cross) if cross else np.array([]),
    }


# ---------------------------------------------------------------------------
# Hybrid: filename parsing + loading
# ---------------------------------------------------------------------------
_HYBRID_RE = re.compile(r"group(\d+)_(F2F|6_1|2_2_2)_(introductions|worstmeal)\.csv$")
HYBRID_CONDITIONS = ["F2F", "6_1", "2_2_2"]
HYBRID_TASK_LABELS = {"introductions": "Introductions (2-person)", "worstmeal": "Worst Meal (3-person)"}


def parse_hybrid_filename(path: Path) -> tuple[str, str, str]:
    m = _HYBRID_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Filename doesn't match group{{N}}_{{condition}}_{{task}}.csv pattern: {path}")
    group_id, condition, task = m.groups()
    return f"group{group_id}", condition, task


def load_hybrid_dataset(dir_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every hybrid CSV. Returns (long_df, specs_df):
      long_df -- unified schema + group_id/condition/task columns, all files concatenated
      specs_df -- one row per file: group_id, condition, task, participants, t0, t1, n_people
    """
    dir_path = Path(dir_path)
    frames, spec_rows = [], []
    for p in sorted(dir_path.glob("group*.csv")):
        group_id, condition, task = parse_hybrid_filename(p)
        df = load_session(p, session_id=f"{group_id}_{condition}_{task}", format="pkl_long_csv")
        df["group_id"], df["condition"], df["task"] = group_id, condition, task
        frames.append(df)
        spec_rows.append({
            "group_id": group_id, "condition": condition, "task": task,
            "participants": sorted(df["participant_id"].unique()),
            "t0": df["t_s"].min(), "t1": df["t_s"].max(),
            "n_people": df["participant_id"].nunique(),
        })
    long_df = pd.concat(frames, ignore_index=True)
    specs_df = pd.DataFrame(spec_rows)
    return long_df, specs_df


def pairwise_distances_within_file(df: pd.DataFrame, rate_hz: float = 10.0) -> np.ndarray:
    """All pairwise 3D distances over time for one already-segmented
    dyad/triad file (all participants in `df` are the task's group)."""
    from xr_analysis import resample_session, all_pairs, HEAD_POS_COLS
    participants = sorted(df["participant_id"].unique())
    if len(participants) < 2:
        return np.array([])
    resampled = resample_session(df, rate_hz=rate_hz)
    dists = []
    for pa, pb in all_pairs(participants):
        a = resampled[pa][HEAD_POS_COLS].to_numpy()
        b = resampled[pb][HEAD_POS_COLS].to_numpy()
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        dists.append(np.linalg.norm(a[valid] - b[valid], axis=1))
    return np.concatenate(dists) if dists else np.array([])


# ---------------------------------------------------------------------------
# Distribution-style proxemics plot (matches the reference figures)
# ---------------------------------------------------------------------------
# (line_color, fill_color) pairs -- navy/lavender first (matches "Face to Face"
# in reference figures), tan/orange second (matches "XR Mediated"), then extra
# pairs for 3+-way comparisons.
DEFAULT_LINE_FILL_COLORS = [
    ("#332a6e", "#b6ade0"),
    ("#d98f3f", "#f4c896"),
    ("#3f8f6e", "#a8dcc6"),
    ("#b23b3b", "#e8a3a3"),
]


def plot_proxemic_distribution(ax, dist_by_label: dict[str, np.ndarray], title: str,
                                colors: Optional[dict] = None, xmax: float = 7.0, bins: int = 70,
                                ymax: Optional[float] = None, caption: Optional[str] = None):
    """Smoothed-histogram line + fill (not bars) per label, Hall-zone boundary
    lines, matching the 'Face to Face vs XR Mediated' reference style."""
    for i, (label, dists) in enumerate(dist_by_label.items()):
        dists = dists[np.isfinite(dists)]
        if len(dists) == 0:
            continue
        line_color, fill_color = (colors or {}).get(label, DEFAULT_LINE_FILL_COLORS[i % len(DEFAULT_LINE_FILL_COLORS)])
        counts, edges = np.histogram(dists, bins=bins, range=(0, xmax), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        # Pad to the axis edges (x=0 and x=xmax) at height 0 so the fill/line
        # touches both edges flush instead of leaving a half-bin-width gap.
        x_plot = np.concatenate([[0.0], centers, [xmax]])
        y_plot = np.concatenate([[0.0], counts, [0.0]])
        ax.plot(x_plot, y_plot, color=line_color, linewidth=1.3, label=label)
        ax.fill_between(x_plot, y_plot, 0, color=fill_color, alpha=0.45, linewidth=0)

    ax.set_xlim(0, xmax)
    ax.margins(x=0)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(bottom=0)
    top = ax.get_ylim()[1]
    n_labels = len(dist_by_label)
    legend_top = 0.93
    legend_bottom = legend_top - 0.09 * max(n_labels, 1) - 0.03  # rough legend box height
    for boundary, label in HALL_BOUNDARIES:
        if boundary <= xmax:
            ax.axvline(boundary, color="black", linewidth=1.0)
            # A legend anchored at upper-right will collide with a boundary
            # label that also falls in the top-right corner (e.g. "Social
            # Distance" when xmax is small) -- drop that label below the
            # legend box instead of behind it.
            in_legend_corner = boundary >= xmax * 0.8
            y_frac = (legend_bottom - 0.04) if in_legend_corner else 0.97
            ax.text(boundary, top * y_frac, label.replace("\n", " "), rotation=90, va="top", ha="right",
                     fontsize=8, color="black")
    ax.set_xlabel("Distance in Meters")
    ax.set_ylabel("Probability Distribution")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, legend_top))
    if caption:
        ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha="center", va="top",
                 fontsize=11, style="italic")


# ---------------------------------------------------------------------------
# Aggregate direction-sync score boxplot (matches the magma/white-median/
# no-spines reference style) -- one box per label, values are per-pair
# corr_agg scores (or any array of scores you hand it).
# ---------------------------------------------------------------------------
def plot_sync_score_boxplot(ax, data_by_label: dict, title: str,
                             ylabel: str = "Aggregate Direction-Sync Score",
                             zero_line: bool = False, rotate_labels: bool = False):
    import matplotlib.pyplot as plt

    labels = list(data_by_label.keys())
    data = []
    for lab in labels:
        arr = np.asarray(data_by_label[lab], dtype=float)
        data.append(arr[np.isfinite(arr)])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    cmap = plt.get_cmap("magma")
    shades = np.linspace(0.25, 0.75, max(len(labels), 2))[:len(labels)] if len(labels) > 1 else [0.4]
    for patch, shade in zip(bp["boxes"], shades):
        patch.set_facecolor(cmap(shade))
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    if zero_line:
        ylim = ax.get_ylim()
        ax.axhline(y=0, color="r", linewidth=1, zorder=0)
        ax.fill_between([0.4, len(labels) + 0.6], 0, ylim[0], color="red", alpha=0.08, zorder=0)
        ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if rotate_labels:
        ax.tick_params(axis="x", rotation=45)


# ---------------------------------------------------------------------------
# Locus of Focus (LOF) export
# ---------------------------------------------------------------------------
# Axis convention swap: our CSV/pkl exports use x/y/z with y = up; Locus of
# Focus expects x/z floor-plane with y/z swapped from our convention.
LOF_RENAME = {
    "position_y": "position_z",
    "position_z": "position_y",
    "direction_y": "direction_z",
    "direction_z": "direction_y",
}


def read_raw_conflab_episode(path: Path) -> pd.DataFrame:
    """Raw (un-normalized, all 62/63 native columns) episode dataframe --
    LOF export needs the full native schema, not xr_loaders' compact unified one."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def read_raw_hybrid_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def export_lof_pickle(raw_df: pd.DataFrame, ref_cols: list[str], out_path: Path) -> pd.DataFrame:
    """Convert one raw native-schema segment (a ConfLab episode or a hybrid
    group/condition/task file -- i.e. a segment where the participants are
    already together as a group) into a Locus-of-Focus-ready pickle: axis
    swap, required placeholder fields, padded/reordered to `ref_cols`."""
    df_lof = raw_df.rename(columns=LOF_RENAME).copy()
    if "uuid" not in df_lof.columns:
        raise ValueError("raw_df must already have a uuid column (native schema)")
    df_lof["display_name"] = df_lof.get("display_name", df_lof["uuid"])
    df_lof["timestamp"] = pd.to_datetime(df_lof["timestamp"], utc=True, format="mixed", errors="coerce")
    df_lof["frame_id"] = df_lof["timestamp"]
    df_lof["volume"] = 0.0
    df_lof["is_speaking"] = False
    df_lof["is_visible"] = True
    df_lof["is_loaded"] = True
    df_lof["is_entered"] = True
    df_lof["is_flying"] = False

    for col in ref_cols:
        if col not in df_lof.columns:
            df_lof[col] = np.nan
    df_lof = df_lof[ref_cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_lof.to_pickle(out_path)
    return df_lof


# ---------------------------------------------------------------------------
# Fashion (Historical / Neutral): filename parsing + loading
# ---------------------------------------------------------------------------
_FASHION_RE = re.compile(r"([A-Z])(\d+)_[A-Z]\d+_session\.csv$")
FASHION_CONDITION_LABEL = {"H": "Historical", "N": "Neutral"}


def parse_fashion_filename(path: Path) -> tuple[str, str]:
    """Returns (pair_id, condition) e.g. 'H01_H02_session.csv' -> ('H01_H02', 'H')."""
    name = Path(path).name
    m = _FASHION_RE.match(name)
    if not m:
        raise ValueError(f"Filename doesn't match {{P1}}_{{P2}}_session.csv pattern: {path}")
    condition = m.group(1)
    pair_id = name[:-len("_session.csv")]
    return pair_id, condition


def load_fashion_dataset(dir_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every fashion dyad-session CSV. Returns (long_df, specs_df):
      long_df -- unified schema + pair_id/condition columns, all sessions concatenated
      specs_df -- one row per file: pair_id, condition, participants, t0, t1, n_people
    """
    dir_path = Path(dir_path)
    frames, spec_rows = [], []
    for p in sorted(dir_path.glob("*_session.csv")):
        pair_id, condition = parse_fashion_filename(p)
        df = load_session(p, session_id=pair_id, format="pkl_long_csv")
        df["pair_id"], df["condition"] = pair_id, condition
        frames.append(df)
        spec_rows.append({
            "pair_id": pair_id, "condition": condition,
            "participants": sorted(df["participant_id"].unique()),
            "t0": df["t_s"].min(), "t1": df["t_s"].max(),
            "n_people": df["participant_id"].nunique(),
        })
    long_df = pd.concat(frames, ignore_index=True)
    specs_df = pd.DataFrame(spec_rows)
    return long_df, specs_df
