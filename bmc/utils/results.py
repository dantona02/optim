"""
results.py
    Utility functions for saving and loading BMCSim simulation results.

    Directory layout (one folder per simulation run):
        results/simulations/{label}_{YYYYMMDD_HHMMSS}/
            metadata.json        – seq_file, config_file, z_positions, date, ...
            m_out.npy            – raw magnetization tensor  (n_iso, n_states, n_time)
            t.npy                – time array
            z_positions.npy      – isochromat positions used
            time_sampling_size.npy
            events.json          – list of simulation events
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from bmc.fid.engine import BMCSim

# Root directory for all simulation results (relative to the repo root)
_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "simulations"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_simulation(
    sim: "BMCSim",
    label: str = "sim",
    results_root: Optional[Path | str] = None,
) -> Path:
    """Save all raw simulation data to a timestamped sub-folder.

    Parameters
    ----------
    sim : BMCSim
        A fully-run BMCSim instance (after ``sim.run_fid()``).
    label : str
        Short human-readable label prepended to the folder name, e.g. ``"fisp_30deg"``.
    results_root : Path | str | None
        Root directory for results.  Defaults to ``results/simulations/`` in the
        repository root.

    Returns
    -------
    Path
        Absolute path to the created output directory.
    """
    root = Path(results_root) if results_root is not None else _DEFAULT_RESULTS_ROOT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"{label}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- raw tensors / arrays ------------------------------------------------
    m_out_np = sim.m_out.detach().cpu().numpy() if isinstance(sim.m_out, torch.Tensor) else np.array(sim.m_out)
    t_np = sim.t.detach().cpu().numpy() if isinstance(sim.t, torch.Tensor) else np.array(sim.t)
    z_pos_np = sim.z_positions.detach().cpu().numpy() if isinstance(sim.z_positions, torch.Tensor) else np.array(sim.z_positions)
    tss_np = sim.time_sampling_size.detach().cpu().numpy() if isinstance(sim.time_sampling_size, torch.Tensor) else np.array(sim.time_sampling_size)

    np.save(out_dir / "m_out.npy", m_out_np)
    np.save(out_dir / "t.npy", t_np)
    np.save(out_dir / "z_positions.npy", z_pos_np)
    np.save(out_dir / "time_sampling_size.npy", tss_np)

    # --- events --------------------------------------------------------------
    with open(out_dir / "events.json", "w") as f:
        json.dump(sim.events, f, indent=2)

    # --- metadata ------------------------------------------------------------
    metadata = {
        "label": label,
        "date": datetime.now().isoformat(),
        "seq_file": str(sim.seq_file),
        "n_isochromats": int(sim.n_isochromats),
        "n_states": int(m_out_np.shape[1]),
        "n_timepoints": int(m_out_np.shape[2]),
        "adc_time": float(sim.adc_time),
        "n_backlog": sim.n_backlog,
        "m_out_shape": list(m_out_np.shape),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- copy config yaml ----------------------------------------------------
    try:
        config_src = Path(sim.seq_file).parent.parent / "sim_lib" / "config_1pool.yaml"
        # Try to get config_file from params if available
        if hasattr(sim, "config_file") and sim.config_file is not None:
            config_src = Path(sim.config_file)
        if config_src.exists():
            shutil.copy2(config_src, out_dir / "config.yaml")
            metadata["config_file"] = str(config_src)
    except Exception:
        pass  # config copy is best-effort

    print(f"[results] Saved simulation to: {out_dir}")
    return out_dir


def load_simulation(path: Path | str) -> dict:
    """Load a previously saved simulation result from disk.

    Parameters
    ----------
    path : Path | str
        Path to the simulation result directory (as returned by ``save_simulation``).

    Returns
    -------
    dict with keys:
        ``m_out``               – np.ndarray (n_iso, n_states, n_time)
        ``t``                   – np.ndarray (n_time,)
        ``z_positions``         – np.ndarray (n_iso,)
        ``time_sampling_size``  – np.ndarray
        ``events``              – list[str]
        ``metadata``            – dict
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result directory not found: {p}")

    result = {
        "m_out": np.load(p / "m_out.npy"),
        "t": np.load(p / "t.npy"),
        "z_positions": np.load(p / "z_positions.npy"),
        "time_sampling_size": np.load(p / "time_sampling_size.npy"),
    }

    with open(p / "events.json") as f:
        result["events"] = json.load(f)

    with open(p / "metadata.json") as f:
        result["metadata"] = json.load(f)

    return result


def list_simulations(results_root: Optional[Path | str] = None) -> list[Path]:
    """Return a sorted list of all saved simulation directories.

    Parameters
    ----------
    results_root : Path | str | None
        Root directory.  Defaults to ``results/simulations/``.

    Returns
    -------
    list[Path]
        Sorted list of simulation directories (newest last).
    """
    root = Path(results_root) if results_root is not None else _DEFAULT_RESULTS_ROOT
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists())
