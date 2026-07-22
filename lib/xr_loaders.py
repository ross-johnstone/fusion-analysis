"""
xr_loaders.py
=============
Format-detecting loaders that normalize different social-XR motion-capture
export formats into one common long-format schema, so downstream analysis
code never has to know which format a session came from.

Supported format families
--------------------------
1. "position_rotation_csv" -- one CSV per participant, semicolon-delimited,
   Euler rotations, e.g. `8-1_position_rotation.csv` / `8-2_position_rotation.csv`.
   Columns: utc_timestamp_ms;participant_id;position_x/y/z;rotation_x/y/z;
   left_hand_position_x/y/z;right_hand_position_x/y/z;
   left_hand_rotation_x/y/z;right_hand_rotation_x/y/z

2. "quest_csv" -- one CSV containing all participants in a session,
   comma-delimited, quaternion rotation, e.g. `Experiment1_30Hz_Quest.csv`.
   Columns: timestamp;device_id;participant_id;headset_id;trial_id;device;
   position_x/y/z;rotation_x/y/z/w (quaternion); forward/backward/right/left/
   up/down vectors; direction_x/y/z. No hand tracking.

3. "pkl_long" -- one pickled pandas DataFrame per session, already long-format
   (multiple participants via a `uuid`/`display_name` column), e.g.
   `H01_H02_fashion_xzy.pkl`, `conflab_ep02_4_33_42.pkl`. Position + direction
   vector only (no rotation quaternion/Euler populated in the examples seen),
   plus optional voice-activity flags (`is_speaking`, `is_muted`, `volume`)
   and presence flags (`is_flying`, `is_visible`, `is_loaded`, `is_entered`).
   No hand tracking.

Unified output schema (one row per participant per tracked frame)
-------------------------------------------------------------------
    session_id        str    session/file identifier
    source_format      str    one of the three family names above
    participant_id      str    stringified participant/device identifier
    t_s              float    unix time in seconds (float)
    position_x/y/z    float    head position (m)
    rotation_x/y/z    float    head rotation, Euler XYZ degrees (NaN if unavailable)
    left_hand_position_x/y/z    float  (NaN if unavailable)
    right_hand_position_x/y/z   float  (NaN if unavailable)
    left_hand_rotation_x/y/z    float  Euler XYZ degrees (NaN if unavailable)
    right_hand_rotation_x/y/z   float  Euler XYZ degrees (NaN if unavailable)
    is_speaking, is_muted        object (NaN if unavailable)

Adding a new format: write a `_load_<name>(path_or_paths) -> pd.DataFrame`
function that returns the unified schema (use `_empty_unified()` as a
starting template) and register it + its detector in `FORMAT_LOADERS` /
`detect_format()`.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

PathLike = Union[str, Path]

UNIFIED_COLUMNS = [
    "session_id", "source_format", "participant_id", "t_s",
    "position_x", "position_y", "position_z",
    "rotation_x", "rotation_y", "rotation_z",
    "left_hand_position_x", "left_hand_position_y", "left_hand_position_z",
    "right_hand_position_x", "right_hand_position_y", "right_hand_position_z",
    "left_hand_rotation_x", "left_hand_rotation_y", "left_hand_rotation_z",
    "right_hand_rotation_x", "right_hand_rotation_y", "right_hand_rotation_z",
    "is_speaking", "is_muted",
]


def _empty_unified(n: int) -> pd.DataFrame:
    return pd.DataFrame({c: [np.nan] * n for c in UNIFIED_COLUMNS})


def _quat_to_euler_deg(qx, qy, qz, qw) -> np.ndarray:
    """(N,4) quaternion arrays -> (N,3) Euler XYZ degrees. Rows with any NaN
    or a near-zero quaternion norm come back as NaN (some exports zero-fill
    untracked hands/objects rather than omitting them)."""
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qz = np.asarray(qz, dtype=float)
    qw = np.asarray(qw, dtype=float)
    quat = np.stack([qx, qy, qz, qw], axis=1)
    norm = np.linalg.norm(quat, axis=1)
    valid = np.isfinite(norm) & (norm > 1e-8)
    out = np.full((len(quat), 3), np.nan)
    if valid.any():
        out[valid] = Rotation.from_quat(quat[valid]).as_euler("xyz", degrees=True)
    return out


# ---------------------------------------------------------------------------
# Format 1: position_rotation_csv (one file per participant, Euler, ';'-sep)
# ---------------------------------------------------------------------------
def _is_position_rotation_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    with open(path, "r") as f:
        header = f.readline()
    return "utc_timestamp_ms" in header and ";" in header


def _load_position_rotation_csv(paths: Sequence[PathLike], session_id: str) -> pd.DataFrame:
    """`paths` is a list of one-file-per-participant CSVs that together make
    up a single session (e.g. [8-1_position_rotation.csv, 8-2_position_rotation.csv])."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        raw = pd.read_csv(p, sep=";")
        n = len(raw)
        out = _empty_unified(n)
        out["session_id"] = session_id
        out["source_format"] = "position_rotation_csv"
        out["participant_id"] = raw["participant_id"].astype(str)
        out["t_s"] = raw["utc_timestamp_ms"].astype(float) / 1000.0
        for c in ["position_x", "position_y", "position_z",
                  "rotation_x", "rotation_y", "rotation_z"]:
            out[c] = raw[c]
        for c in ["left_hand_position_x", "left_hand_position_y", "left_hand_position_z",
                  "right_hand_position_x", "right_hand_position_y", "right_hand_position_z",
                  "left_hand_rotation_x", "left_hand_rotation_y", "left_hand_rotation_z",
                  "right_hand_rotation_x", "right_hand_rotation_y", "right_hand_rotation_z"]:
            if c in raw.columns:
                out[c] = raw[c]
        # (0, 0, 0) is this format's "hand not tracked yet" sentinel (visible
        # in the first frame of each file) -- treat as missing, not real data.
        for side in ("left", "right"):
            pos_cols = [f"{side}_hand_position_{a}" for a in "xyz"]
            rot_cols = [f"{side}_hand_rotation_{a}" for a in "xyz"]
            zeroed = (out[pos_cols] == 0).all(axis=1) & (out[rot_cols] == 0).all(axis=1)
            out.loc[zeroed, pos_cols + rot_cols] = np.nan
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Format 2: quest_csv (one file, whole session, comma-sep, quaternion)
# ---------------------------------------------------------------------------
def _is_quest_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    with open(path, "r") as f:
        header = f.readline()
    return "headset_id" in header and "rotation_w" in header


def _load_quest_csv(path: PathLike, session_id: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    n = len(raw)
    out = _empty_unified(n)
    out["session_id"] = session_id
    out["source_format"] = "quest_csv"
    out["participant_id"] = raw["participant_id"].astype(str)
    out["t_s"] = pd.to_datetime(raw["timestamp"], utc=True, format="mixed").astype("int64") / 1e9
    for c in ["position_x", "position_y", "position_z"]:
        out[c] = raw[c]
    euler = _quat_to_euler_deg(raw["rotation_x"], raw["rotation_y"],
                                raw["rotation_z"], raw["rotation_w"])
    out[["rotation_x", "rotation_y", "rotation_z"]] = euler
    # No hand tracking in this format -- left as NaN.
    if "trial_id" in raw.columns:
        out["session_id"] = session_id + "_trial" + raw["trial_id"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Format 3: pkl_long (pre-pickled long-format pandas DataFrame)
# ---------------------------------------------------------------------------
def _is_pkl_long(path: Path) -> bool:
    return path.suffix.lower() == ".pkl"


def _normalize_pkl_long_df(raw: pd.DataFrame, session_id: str, source_format: str) -> pd.DataFrame:
    raw = raw.dropna(subset=["uuid", "timestamp"]).reset_index(drop=True)
    n = len(raw)
    out = _empty_unified(n)
    out["session_id"] = session_id
    out["source_format"] = source_format
    out["participant_id"] = raw["uuid"].astype(str)
    out["t_s"] = pd.to_datetime(raw["timestamp"], utc=True, format="mixed").astype("int64") / 1e9
    for c in ["position_x", "position_y", "position_z"]:
        if c in raw.columns:
            out[c] = raw[c]
    # Head rotation: prefer a populated quaternion field if one exists in
    # this particular export; otherwise leave as NaN (the fashion/conflab
    # examples only populate direction_x/y/z, a facing unit-vector, not a
    # full rotation -- not enough to reconstruct Euler angles).
    for quat_cols, label in [
        (("position_quatx", "position_quaty", "position_quatz", "position_quatw,"), "position_quat"),
        (("rig_quat_x", "rig_quat_y", "rig_quat_z", "rig_quat_w"), "rig_quat"),
    ]:
        if all(c in raw.columns for c in quat_cols) and raw[quat_cols[0]].notna().any():
            euler = _quat_to_euler_deg(*[raw[c] for c in quat_cols])
            out[["rotation_x", "rotation_y", "rotation_z"]] = euler
            break
    for side, prefix in [("left", "lefthand"), ("right", "righthand")]:
        pos_cols = [f"{prefix}_position{a}" for a in "xyz"]
        quat_cols = [f"{prefix}_quat{a}" for a in "XYZ"] + [f"{prefix}_quatW"]
        if all(c in raw.columns for c in pos_cols) and raw[pos_cols[0]].notna().any():
            out[[f"{side}_hand_position_x", f"{side}_hand_position_y", f"{side}_hand_position_z"]] = raw[pos_cols].to_numpy()
        if all(c in raw.columns for c in quat_cols) and raw[quat_cols[0]].notna().any():
            euler = _quat_to_euler_deg(*[raw[c] for c in quat_cols])
            out[[f"{side}_hand_rotation_x", f"{side}_hand_rotation_y", f"{side}_hand_rotation_z"]] = euler
    for c in ["is_speaking", "is_muted"]:
        if c in raw.columns:
            out[c] = raw[c]
    return out


def _load_pkl_long(path: PathLike, session_id: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, pd.DataFrame):
        raise ValueError(f"{path}: expected a pickled pandas DataFrame, got {type(raw)}")
    return _normalize_pkl_long_df(raw, session_id, "pkl_long")


def _is_pkl_long_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    with open(path, "r") as f:
        header = f.readline()
    # Same native schema as pkl_long, just exported as CSV (e.g. the "hybrid"
    # dataset and full-session ConfLab exports) -- distinguish via columns
    # that are unique to this schema and absent from the other two CSV formats.
    return "rig_posx" in header and "uuid" in header and "," in header


def _load_pkl_long_csv(path: PathLike, session_id: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return _normalize_pkl_long_df(raw, session_id, "pkl_long_csv")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
FORMAT_DETECTORS = {
    "position_rotation_csv": _is_position_rotation_csv,
    "quest_csv": _is_quest_csv,
    "pkl_long_csv": _is_pkl_long_csv,
    "pkl_long": _is_pkl_long,
}


def detect_format(path: PathLike) -> str:
    path = Path(path)
    for name, test in FORMAT_DETECTORS.items():
        if test(path):
            return name
    raise ValueError(f"Could not detect a known format for {path}")


def load_session(paths: Union[PathLike, Sequence[PathLike]], session_id: str = None,
                  format: str = None) -> pd.DataFrame:
    """Load one session (one or more files that together represent a single
    recording) into the unified long-format schema.

    `paths` -- a single file for self-contained formats (quest_csv, pkl_long),
    or a list of one-file-per-participant CSVs for position_rotation_csv.
    `session_id` -- defaults to the stem of the first file.
    `format` -- override auto-detection ('position_rotation_csv' | 'quest_csv' | 'pkl_long').
    """
    is_multi = isinstance(paths, (list, tuple))
    first_path = Path(paths[0] if is_multi else paths)
    fmt = format or detect_format(first_path)
    session_id = session_id or first_path.stem

    if fmt == "position_rotation_csv":
        df = _load_position_rotation_csv(paths if is_multi else [paths], session_id)
    elif fmt == "quest_csv":
        df = _load_quest_csv(first_path, session_id)
    elif fmt == "pkl_long":
        df = _load_pkl_long(first_path, session_id)
    elif fmt == "pkl_long_csv":
        df = _load_pkl_long_csv(first_path, session_id)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    df = df.sort_values(["participant_id", "t_s"]).reset_index(drop=True)
    return df[UNIFIED_COLUMNS]


def load_sessions(specs: Iterable[dict]) -> pd.DataFrame:
    """Load and concatenate multiple sessions.
    `specs` -- iterable of dicts, each with keys matching `load_session`'s
    kwargs, e.g. [{"paths": "a.csv", "session_id": "s1"}, ...].
    """
    return pd.concat([load_session(**spec) for spec in specs], ignore_index=True)
