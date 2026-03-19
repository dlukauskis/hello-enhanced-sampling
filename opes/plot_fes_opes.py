"""
OPES post-simulation analysis & plotting.
Produces the same 4-panel figure as wt-metadynamics/plot_fes.py:

  (1) Phi / Psi evolution over time
  (2) 2D Free Energy Landscape (final snapshot)
  (3) 1D Free Energy Profile projected along phi (multiple snapshots)
  (4) Free Energy Difference ΔF(α→β) over simulation time
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from analysis_opes import (
    read_cv_history,
    load_opes_kernel_snapshots,
    reconstruct_fes_opes,
    project_fes_1d,
)


# Phi / Psi basin index ranges on a 200-point grid from -π to π
# Alpha-helix region: φ ∈ [-π/2, -π/4]  → indices [50, 75)
# Beta-sheet  region: φ ∈ [0,    3π/4]  → indices [100, 175)
_ALPHA_SLICE = slice(50, 75)
_BETA_SLICE  = slice(100, 175)


def main(
    output_dir='output',
    cv_history_file=None,
    clip_fes_to=50,           # kJ/mol – colour-scale cap for 2D FES
    fig_fname='opes-aladipep-results.png',
    n_fes_snapshots=5,        # curves shown in the 1D-FES panel
    max_snapshots=100,        # cap on total snapshots loaded (limits compute)
    periodic=None,            # periodic ranges; defaults to [(-π,π),(-π,π)]
    kT=2.494,                 # 300 K in kJ/mol
):
    """
    Build and save the 4-panel results figure.

    Parameters
    ----------
    output_dir : str
        Directory containing ``cv_history.txt`` and ``kernels_<step>.npz``.
    cv_history_file : str or None
        Path to cv_history.txt; defaults to ``<output_dir>/cv_history.txt``.
    clip_fes_to : float
        Upper limit (kJ/mol) for the 2D FES colour bar.
    fig_fname : str or None
        File name to save the figure.  Pass ``None`` to skip saving.
    n_fes_snapshots : int
        Number of 1D-FES curves to overlay in panel 3.
    max_snapshots : int
        Maximum kernel snapshots to load for the ΔF convergence trace.
    periodic : list of (float, float) or None
        Periodic CV ranges.  Defaults to ``[(-π, π), (-π, π)]``.
    kT : float
        Thermal energy in kJ/mol (default 2.494 for 300 K).
    """
    if cv_history_file is None:
        cv_history_file = os.path.join(output_dir, 'cv_history.txt')
    if periodic is None:
        periodic = [(-np.pi, np.pi), (-np.pi, np.pi)]

    # ------------------------------------------------------------------
    # 1. CV time series
    # ------------------------------------------------------------------
    print(f"Reading CV history from {cv_history_file} …")
    time_ps, cv_arrays = read_cv_history(cv_history_file)
    print(f"  {len(time_ps)} frames")

    # ------------------------------------------------------------------
    # 2. Load kernel snapshots (all of them, up to max_snapshots)
    # ------------------------------------------------------------------
    print(f"Loading kernel snapshots from {output_dir} …")
    snapshots = load_opes_kernel_snapshots(output_dir, max_snapshots=max_snapshots)
    print(f"  {len(snapshots)} snapshots loaded")

    if not snapshots:
        print("No valid kernel snapshots found – plotting CV history only.")
        _plot_cv_only(time_ps, cv_arrays, fig_fname)
        return

    # ------------------------------------------------------------------
    # 3. Reconstruct FES for every snapshot → deltaF convergence trace
    # ------------------------------------------------------------------
    cv_grid = [np.linspace(-np.pi, np.pi, 200), np.linspace(-np.pi, np.pi, 200)]

    steps_all, delta_F_all, fes_all = [], [], []
    print("Reconstructing FES for each snapshot …")
    for i, snap in enumerate(snapshots):
        fes = reconstruct_fes_opes(
            snap['kernels'],
            snap['sigma_vals'],
            snap['sum_weights'],
            snap['kT'],
            cv_grid,
            periodic=periodic,
            barrier=snap.get('barrier'),
            bias_factor=snap.get('bias_factor'),
            kernel_cutoff=snap.get('kernel_cutoff'),
        )
        fes_all.append(fes)

        _, fes_phi = project_fes_1d(cv_grid[0], cv_grid[1], fes, kT=kT, project_along='cv0')
        alpha_fe = np.min(fes_phi[_ALPHA_SLICE])
        beta_fe  = np.min(fes_phi[_BETA_SLICE])
        delta_F_all.append(alpha_fe - beta_fe)
        steps_all.append(snap['step'])

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(snapshots)}")

    # Timestep is 2 fs, so step → ps
    time_ps_snaps = [s * 0.002 for s in steps_all]

    # ------------------------------------------------------------------
    # 4. Pick evenly-spaced subset for the 1D-FES overlay panel
    # ------------------------------------------------------------------
    if len(fes_all) <= n_fes_snapshots:
        display_idx = list(range(len(fes_all)))
    else:
        display_idx = list(np.linspace(0, len(fes_all) - 1, n_fes_snapshots, dtype=int))

    # ------------------------------------------------------------------
    # 5. Build the figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flatten()

    # --- Panel 1: CV evolution ---
    ax0.scatter(time_ps, cv_arrays[0], s=6, alpha=0.5, label='phi (CV0)')
    if len(cv_arrays) > 1:
        ax0.scatter(time_ps, cv_arrays[1], s=6, alpha=0.5, label='psi (CV1)')
    ax0.set_xlabel('Time (ps)')
    ax0.set_ylabel('CV values (rad)')
    ax0.set_title('Phi and Psi Evolution Over Time')
    ax0.legend(markerscale=2)

    # --- Panel 2: 2D FES (final snapshot) ---
    final_fes = fes_all[-1]
    im = ax1.imshow(
        final_fes,
        extent=(-np.pi, np.pi, -np.pi, np.pi),
        origin='lower',
        cmap='viridis',
        aspect='auto',
        vmin=0,
        vmax=clip_fes_to,
    )
    ax1.set_xlabel('Phi (rad)')
    ax1.set_ylabel('Psi (rad)')
    ax1.set_title('Free Energy Landscape of Alanine Dipeptide in vacuo (OPES)')
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Free Energy (kJ/mol)')

    # --- Panel 3: 1D FES snapshots ---
    ax2.set_title('1D Free Energy Profile (Projected Along Phi)')
    for idx in display_idx:
        fes_snap = fes_all[idx]
        phi_grid, fes_phi = project_fes_1d(
            cv_grid[0], cv_grid[1], fes_snap, kT=kT, project_along='cv0'
        )
        label = f'{time_ps_snaps[idx]:.0f} ps'
        ax2.plot(phi_grid, fes_phi, linewidth=2, label=label)
    ax2.set_xlabel('Phi (rad)')
    ax2.set_ylabel('Free Energy (kJ/mol)')
    ax2.legend(fontsize=8)

    # --- Panel 4: ΔF(α→β) convergence ---
    final_dF = delta_F_all[-1] if delta_F_all else float('nan')
    ax3.plot(time_ps_snaps, delta_F_all, 'r-', linewidth=2,
             label=f'final ΔF = {final_dF:.2f} kJ/mol')
    ax3.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('ΔF (kJ/mol)')
    ax3.set_title('Free Energy Difference Between Alpha and Beta Basins')
    ax3.legend()

    if fig_fname:
        plt.savefig(fig_fname, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {fig_fname}")

    plt.show()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _plot_cv_only(time_ps, cv_arrays, fig_fname):
    """Minimal fallback: just CV time series when no kernel data is available."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.scatter(time_ps, cv_arrays[0], s=6, alpha=0.5, label='phi')
    if len(cv_arrays) > 1:
        ax.scatter(time_ps, cv_arrays[1], s=6, alpha=0.5, label='psi')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('CV (rad)')
    ax.set_title('CV Evolution (no kernel data found)')
    ax.legend(markerscale=2)
    if fig_fname:
        plt.savefig(fig_fname, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

