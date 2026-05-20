"""
Worker for the isochromat-convergence sweep.

Must live in a module (not a notebook cell) so multiprocessing can pickle it.
Each worker process is fully independent: it loads params, builds BMCSim, runs,
extracts the echo amplitude, and returns a lightweight tuple.

n_backlog=1 is used instead of 'ALL' to keep per-process memory low.
For a 5-block Hahn-echo sequence this records the last two blocks
(final TE/2 delay + ADC), which always contains the echo peak.
"""
import numpy as np
import torch


def run_convergence_sim(args: tuple) -> tuple:
    """
    Simulate one (mode, N) point for the convergence sweep.

    Parameters
    ----------
    args : tuple
        (config_file, seq_file, mode, n_iso, adc_time, t_echo_s, search_win_s)
        All scalars / strings so the tuple is picklable without restriction.

    Returns
    -------
    (mode, n_iso, amp, error_str)
        error_str is None on success, a string on failure.
    """
    config_file, seq_file, mode, n_iso, adc_time, t_echo_s, search_win_s = args

    try:
        from bmc.set_params import load_params
        from bmc.fid.engine import BMCSim

        params = load_params(config_file)
        if mode == "deterministic":
            params.update_options(isochromat_mode="deterministic")

        z = torch.linspace(-1e-3, 1e-3, int(n_iso))
        sim = BMCSim(
            adc_time, params, seq_file,
            z_positions=z, n_backlog=1, webhook=False, verbose=False,
        )
        sim.run_fid()

        t_raw, _, _, _, m_c_total = sim.get_mag()
        t_s = t_raw.cpu().numpy()
        mxy = torch.abs(m_c_total).cpu().numpy()

        mask = np.abs(t_s - t_echo_s) < search_win_s
        amp = float(mxy[mask].max()) if mask.sum() > 0 \
              else float(mxy[np.argmin(np.abs(t_s - t_echo_s))])

        del sim
        return mode, int(n_iso), amp, None

    except Exception as exc:
        return mode, int(n_iso), None, str(exc)
