"""
xr_analysis.py
===============
Group-size-agnostic (N-participant) analyses on top of the unified schema
produced by `xr_loaders.py`: resampling to a shared time grid, temporal
synchrony, social synchrony, hand-movement synchrony, and proxemics.

Every analysis that used to be hard-coded for a dyad (Person A / Person B)
now runs over *all pairs* of participants present in a session and returns
either a long-format per-pair table or a participant x participant summary
matrix, so a 2-person and a 7-person session go through the same code path.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial.distance

HEAD_POS_COLS = ["position_x", "position_y", "position_z"]
DIRECTION_COLS = ["direction_x", "direction_y", "direction_z"]
HAND_COLS = {
    "left": ["left_hand_position_x", "left_hand_position_y", "left_hand_position_z"],
    "right": ["right_hand_position_x", "right_hand_position_y", "right_hand_position_z"],
}

HALL_ZONES = [
    (0.00, 0.46, "intimate"),
    (0.46, 1.20, "personal"),
    (1.20, 3.60, "social"),
    (3.60, np.inf, "public"),
]


# ---------------------------------------------------------------------------
# Native sample-rate estimation -- so the notebook doesn't have to hard-code
# a frame rate per format; it just asks the data.
# ---------------------------------------------------------------------------
def estimate_native_hz(df: pd.DataFrame) -> float:
    """Median per-participant sampling rate (Hz), robust to jitter/dropped
    frames. Works for any format since it only looks at t_s + participant_id."""
    rates = []
    for p, d in df.groupby("participant_id"):
        dt = d.sort_values("t_s")["t_s"].diff().dropna()
        dt = dt[dt > 0]
        if len(dt):
            rates.append(1.0 / dt.median())
    return float(np.median(rates)) if rates else np.nan


# ---------------------------------------------------------------------------
# Resampling -- put every participant in a session onto one shared time grid
# ---------------------------------------------------------------------------
def resample_session(df: pd.DataFrame, rate_hz: float) -> dict[str, pd.DataFrame]:
    """`df` is one session in the unified schema (>= 1 participants).
    Returns {participant_id: resampled_df} where every resampled_df shares
    the same `session_time_s` grid, spanning the *overlap* of all
    participants' tracked time ranges. Position columns are always
    interpolated; hand columns are interpolated only for participants that
    have any hand data, otherwise left as NaN (so hand-based analyses can
    cleanly skip participants without hand tracking)."""
    participants = sorted(df["participant_id"].unique())
    t0 = max(df.loc[df["participant_id"] == p, "t_s"].min() for p in participants)
    t1 = min(df.loc[df["participant_id"] == p, "t_s"].max() for p in participants)
    if t1 <= t0:
        raise ValueError("Participants' tracked time ranges do not overlap.")
    grid = np.arange(t0, t1, 1.0 / rate_hz)

    interp_cols = HEAD_POS_COLS + DIRECTION_COLS + HAND_COLS["left"] + HAND_COLS["right"]
    out = {}
    for p in participants:
        d = (df[df["participant_id"] == p]
             .sort_values("t_s").drop_duplicates("t_s").set_index("t_s"))
        resampled = pd.DataFrame(index=grid)
        resampled.index.name = "t_s"
        for c in interp_cols:
            valid = d[c].notna()
            if valid.sum() < 2:
                resampled[c] = np.nan
            else:
                resampled[c] = np.interp(grid, d.index[valid].to_numpy(),
                                          d.loc[valid, c].to_numpy())
        resampled = resampled.reset_index()
        resampled["session_time_s"] = resampled["t_s"] - t0
        out[p] = resampled
    return out


def has_hand_data(resampled_p: pd.DataFrame) -> bool:
    cols = HAND_COLS["left"] + HAND_COLS["right"]
    return resampled_p[cols].notna().any().any()


def has_direction_data(resampled_p: pd.DataFrame) -> bool:
    return resampled_p[DIRECTION_COLS].notna().any().any()


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------
def _frame_distance_series(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Frame-to-frame Euclidean distance (motion magnitude) for the given
    position columns, NaN-safe (NaN in either frame -> NaN distance)."""
    pos = df[cols].to_numpy()
    d = np.full(len(pos), np.nan)
    diffs = np.diff(pos, axis=0)
    d[1:] = np.linalg.norm(diffs, axis=1)
    return d


def _hand_energy_series(df: pd.DataFrame) -> np.ndarray:
    """Per-frame hand-motion magnitude, averaging whichever of the two hands
    has data (so a participant tracked on only one hand still contributes)."""
    energies = []
    for side, cols in HAND_COLS.items():
        if df[cols].notna().any().any():
            energies.append(_frame_distance_series(df, cols))
    if not energies:
        return np.full(len(df), np.nan)
    stacked = np.vstack(energies)
    counts = np.sum(~np.isnan(stacked), axis=0)
    sums = np.nansum(stacked, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return result


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a, b = a[mask], b[mask]
    da, db = a - a.mean(), b - b.mean()
    den_a, den_b = np.sum(da * da), np.sum(db * db)
    if den_a == 0 or den_b == 0:
        return np.nan
    return float(np.sum(da * db) / np.sqrt(den_a * den_b))


def normalize_sync_score(r: float) -> float:
    return (r + 1.0) / 2.0 if np.isfinite(r) else np.nan


def all_pairs(participant_ids: list[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(participant_ids), 2))


# ---------------------------------------------------------------------------
# 1. Temporal synchrony (all pairs) -- windowed correlation of 3D head speed
# ---------------------------------------------------------------------------
def temporal_synchrony(resampled: dict[str, pd.DataFrame], fps: float,
                        window_s: float = 1.0) -> pd.DataFrame:
    """Returns a long df: participant_a, participant_b, session_time_s,
    pearson_r, synchrony (normalized to [0,1]) -- one row per pair per window."""
    win = max(int(window_s * fps), 1)
    speed = {p: _frame_distance_series(d, HEAD_POS_COLS) for p, d in resampled.items()}
    any_p = next(iter(resampled))
    t = resampled[any_p]["session_time_s"].to_numpy()
    n_windows = len(t) // win

    rows = []
    for pa, pb in all_pairs(list(resampled.keys())):
        sa, sb = speed[pa], speed[pb]
        for i in range(n_windows):
            sl = slice(i * win, (i + 1) * win)
            r = pearson_safe(sa[sl], sb[sl])
            rows.append({"participant_a": pa, "participant_b": pb,
                         "session_time_s": t[sl][0], "pearson_r": r,
                         "synchrony": normalize_sync_score(r)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Social synchrony (all pairs) -- windowed correlation of horizontal
#    (x, z) movement amplitude; returns a distribution per pair
# ---------------------------------------------------------------------------
def social_synchrony(resampled: dict[str, pd.DataFrame], fps: float,
                      window_s: float = 1.0) -> pd.DataFrame:
    """Returns a long df: participant_a, participant_b, pearson_r -- one row
    per pair per window (the distribution IS the return value; aggregate
    with .groupby(['participant_a','participant_b'])['pearson_r'] downstream)."""
    win = int(window_s * fps)
    amp = {p: np.sqrt(d["position_x"].diff() ** 2 + d["position_z"].diff() ** 2).to_numpy()
           for p, d in resampled.items()}

    rows = []
    for pa, pb in all_pairs(list(resampled.keys())):
        a, b = amp[pa][1:], amp[pb][1:]
        n = min(len(a), len(b))
        if n < win * 2:
            continue
        for start in range(0, n - win, win):
            sa, sb = a[start:start + win], b[start:start + win]
            if not (np.isfinite(sa).all() and np.isfinite(sb).all()):
                continue
            if np.std(sa) < 1e-6 or np.std(sb) < 1e-6:
                continue
            r, _ = scipy.stats.pearsonr(sa, sb)
            if np.isfinite(r):
                rows.append({"participant_a": pa, "participant_b": pb, "pearson_r": r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2b. Aggregate direction-sync score (all pairs) -- ONE score per pair, not
#     windowed. Adapted from a validated methodology: align two participants
#     on a shared time base, take the frame-to-frame delta of their facing
#     *direction* vector (not position), and Pearson-correlate each axis's
#     delta series between the pair. corr_agg = corr_x + corr_y + corr_z.
#     This is what "social synchrony" reports in the notebooks now --
#     temporal_synchrony (above) is untouched.
# ---------------------------------------------------------------------------
def direction_sync_score(dir_a: np.ndarray, dir_b: np.ndarray, min_frames: int = 5) -> dict:
    """`dir_a`, `dir_b` -- (N, 3) direction-vector arrays for two participants,
    already on the same shared time grid (e.g. from `resample_session`).
    Returns {'corr_agg', 'corr_x', 'corr_y', 'corr_z', 'n_frames'}."""
    valid = np.isfinite(dir_a).all(axis=1) & np.isfinite(dir_b).all(axis=1)
    a, b = dir_a[valid], dir_b[valid]
    if len(a) < min_frames:
        return {"corr_agg": np.nan, "corr_x": np.nan, "corr_y": np.nan, "corr_z": np.nan, "n_frames": len(a)}
    delta_a = np.diff(a, axis=0)
    delta_b = np.diff(b, axis=0)
    corr_x = pearson_safe(delta_a[:, 0], delta_b[:, 0])
    corr_y = pearson_safe(delta_a[:, 1], delta_b[:, 1])
    corr_z = pearson_safe(delta_a[:, 2], delta_b[:, 2])
    per_axis = np.array([corr_x, corr_y, corr_z])
    # Sums the defined axes (nansum, not a plain sum): some formats only track
    # a floor-plane facing direction (direction_y is identically 0 -> zero
    # variance -> undefined correlation on that axis), and a plain sum would
    # make corr_agg NaN for every pair in those datasets. An axis with real
    # signal in both formats we've seen (fashion, quest) still contributes
    # normally; nansum only changes behavior for axes that carry no signal.
    corr_agg = np.nansum(per_axis) if np.isfinite(per_axis).any() else np.nan
    return {"corr_agg": corr_agg, "corr_x": corr_x, "corr_y": corr_y, "corr_z": corr_z, "n_frames": len(a)}


def direction_sync_all_pairs(resampled: dict[str, pd.DataFrame], min_frames: int = 5) -> pd.DataFrame:
    """All-pairs generalization: one row per pair with corr_agg/corr_x/corr_y/corr_z.
    Pairs where either participant has no direction data at all are skipped
    (not an error -- direction/facing vectors are format-dependent, same
    pattern as hand tracking)."""
    dir_havers = [p for p, d in resampled.items() if has_direction_data(d)]
    rows = []
    for pa, pb in all_pairs(dir_havers):
        dir_a = resampled[pa][DIRECTION_COLS].to_numpy()
        dir_b = resampled[pb][DIRECTION_COLS].to_numpy()
        result = direction_sync_score(dir_a, dir_b, min_frames=min_frames)
        rows.append({"participant_a": pa, "participant_b": pb, **result})
    return pd.DataFrame(rows, columns=["participant_a", "participant_b", "corr_agg", "corr_x", "corr_y", "corr_z", "n_frames"])


# ---------------------------------------------------------------------------
# 3. Hand-movement synchrony (all pairs with hand data) -- windowed
#    correlation of combined left/right hand motion magnitude
# ---------------------------------------------------------------------------
def hand_synchrony(resampled: dict[str, pd.DataFrame], fps: float,
                    window_s: float = 1.0) -> pd.DataFrame:
    """Same pattern as temporal_synchrony but on hand-motion energy instead
    of head position. Pairs where either participant has no hand data at all
    are skipped (not an error -- hand tracking is format-dependent)."""
    win = max(int(window_s * fps), 1)
    hand_havers = [p for p, d in resampled.items() if has_hand_data(d)]
    energy = {p: _hand_energy_series(resampled[p]) for p in hand_havers}
    if len(hand_havers) < 2:
        return pd.DataFrame(columns=["participant_a", "participant_b",
                                      "session_time_s", "pearson_r", "synchrony"])
    any_p = hand_havers[0]
    t = resampled[any_p]["session_time_s"].to_numpy()
    n_windows = len(t) // win

    rows = []
    for pa, pb in all_pairs(hand_havers):
        sa, sb = energy[pa], energy[pb]
        for i in range(n_windows):
            sl = slice(i * win, (i + 1) * win)
            r = pearson_safe(sa[sl], sb[sl])
            rows.append({"participant_a": pa, "participant_b": pb,
                         "session_time_s": t[sl][0], "pearson_r": r,
                         "synchrony": normalize_sync_score(r)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Proxemics (all pairs, naturally N-participant via pdist)
# ---------------------------------------------------------------------------
def compute_proxemics(resampled: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Returns a long df: session_time_s, participant_a, participant_b,
    distance_m -- pairwise 3D head distance at every shared timestamp."""
    participants = sorted(resampled.keys())
    any_p = participants[0]
    t = resampled[any_p]["session_time_s"].to_numpy()
    positions = np.stack([resampled[p][HEAD_POS_COLS].to_numpy() for p in participants], axis=1)
    # positions.shape == (n_frames, n_participants, 3)

    pairs = all_pairs(participants)
    rows = {("session_time_s",): t}
    records = []
    for frame_idx in range(positions.shape[0]):
        xyz = positions[frame_idx]
        if np.isnan(xyz).any():
            dists = [np.nan] * len(pairs)
        else:
            dists = scipy.spatial.distance.pdist(xyz, "euclidean")
        for (pa, pb), dist in zip(pairs, dists):
            records.append((t[frame_idx], pa, pb, dist))
    return pd.DataFrame(records, columns=["session_time_s", "participant_a", "participant_b", "distance_m"])


def zone_breakdown(distances: np.ndarray) -> dict[str, float]:
    distances = distances[np.isfinite(distances)]
    n = len(distances)
    if n == 0:
        return {label: np.nan for _, _, label in HALL_ZONES}
    return {label: 100.0 * ((distances >= lo) & (distances < hi)).sum() / n
            for lo, hi, label in HALL_ZONES}


# ---------------------------------------------------------------------------
# Summary matrices (participant x participant), handy for heatmaps
# ---------------------------------------------------------------------------
def pair_summary_matrix(pair_df: pd.DataFrame, value_col: str,
                         participants: list[str], agg: str = "mean") -> pd.DataFrame:
    """Turn a long per-pair table (participant_a, participant_b, value_col)
    into a symmetric participant x participant summary matrix."""
    mat = pd.DataFrame(np.nan, index=participants, columns=participants, dtype=float)
    if len(pair_df):
        summarized = pair_df.groupby(["participant_a", "participant_b"])[value_col].agg(agg)
        for (pa, pb), val in summarized.items():
            mat.loc[pa, pb] = val
            mat.loc[pb, pa] = val
    np.fill_diagonal(mat.values, np.nan)
    return mat
