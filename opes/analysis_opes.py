"""
Analysis utilities for OPES simulations.
Analogous to wt-metadynamics/analysis.py but for OPES kernel/CV data.
"""
import math
import glob
import os

import numpy as np


def read_cv_history(filename):
    """
    Read the cv_history.txt written by OPESCVReporter.

    Returns
    -------
    time_ps : ndarray, shape (N,)
    cv_arrays : list of ndarray, each shape (N,)
        One array per CV (cv0=phi, cv1=psi, …)
    """
    data = np.loadtxt(filename, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    time_ps = data[:, 0]
    cv_arrays = [data[:, i + 1] for i in range(data.shape[1] - 1)]
    return time_ps, cv_arrays


def _sigma_from_height(height, n_cvs, equal_sigma=True):
    """
    Recover sigma from the stored Gaussian height.
    height = 1 / [(2π)^(d/2) * Π σ_i]
    Assumes equal sigmas when equal_sigma=True:
        σ = [1 / (height * (2π)^(d/2))]^(1/d)
    """
    sigma = (1.0 / (height * (2.0 * math.pi) ** (n_cvs / 2.0))) ** (1.0 / n_cvs)
    return [sigma] * n_cvs


def load_opes_kernel_snapshots(output_dir, max_snapshots=None):
    """
    Load OPES kernel snapshots from *output_dir*.

    Files must be named ``kernels_<step>.npz``.  If the npz was written by the
    updated ``saveKernels`` it contains ``sigma_vals``; otherwise sigma is
    inferred from the stored Gaussian heights (equal-sigma assumption).

    Parameters
    ----------
    output_dir : str
    max_snapshots : int or None
        Cap the number of snapshots (evenly spaced) to avoid slow FES
        reconstruction.  ``None`` keeps all.

    Returns
    -------
    list of dict, each with keys:
        step, kernels (ndarray), sigma_vals (list), sum_weights (float), kT (float)
    """
    pattern = os.path.join(output_dir, 'kernels_*.npz')
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.basename(f).replace('kernels_', '').replace('.npz', ''))
    )

    if not files:
        return []

    # Subsample evenly if requested
    if max_snapshots is not None and len(files) > max_snapshots:
        indices = np.linspace(0, len(files) - 1, max_snapshots, dtype=int)
        files = [files[i] for i in indices]

    snapshots = []
    for fpath in files:
        step = int(os.path.basename(fpath).replace('kernels_', '').replace('.npz', ''))
        data = np.load(fpath, allow_pickle=True)

        kernels = data['kernels']
        if len(kernels) == 0:
            continue

        n_cvs = kernels.shape[1] - 2  # columns: cv0, cv1, ..., weight, height

        # sigma_vals – new-format files store it directly
        if 'sigma_vals' in data:
            sigma_vals = data['sigma_vals'].tolist()
        else:
            # Back-compat: derive from the first height value (equal-sigma)
            height_sample = float(kernels[0, n_cvs + 1])
            sigma_vals = _sigma_from_height(height_sample, n_cvs)

        # sum_weights
        if 'sum_weights' in data:
            sum_weights = float(data['sum_weights'])
        else:
            sum_weights = float(kernels[:, n_cvs].sum())

        # kT
        if 'kT' in data:
            kT = float(data['kT'])
        else:
            kT = 2.494  # 300 K default

        snapshots.append({
            'step': step,
            'kernels': kernels,
            'sigma_vals': sigma_vals,
            'sum_weights': sum_weights,
            'kT': kT,
        })

    return snapshots


def reconstruct_fes_opes(kernels, sigma_vals, sum_weights, kT, cv_grid, periodic=None):
    """
    Reconstruct the FES from a set of OPES kernels on a 2-CV grid.

    Uses **xy** meshgrid convention so the returned array has shape
    ``(n_cv1, n_cv0)`` – identical to the wt-metaD FES convention and
    directly compatible with :func:`project_fes_1d`.

    Parameters
    ----------
    kernels : ndarray, shape (K, n_cvs + 2)
        [cv0, cv1, …, weight, height] for each kernel.
    sigma_vals : list of float
        Bandwidth per CV.
    sum_weights : float
        Sum of kernel weights (normalisation denominator).
    kT : float
        Thermal energy in kJ/mol.
    cv_grid : list of ndarray
        Grid points for each CV, e.g. ``[phi_grid, psi_grid]``.
    periodic : list of (float, float) or None
        Periodic range per CV; ``None`` entries mean non-periodic.

    Returns
    -------
    fes : ndarray, shape (n_cv1, n_cv0)
        Zero-shifted free energy surface in kJ/mol.
    """
    n_cvs = len(sigma_vals)
    if periodic is None:
        periodic = [None] * n_cvs

    # xy meshgrid: shape (n_cv1, n_cv0)
    CV0, CV1 = np.meshgrid(cv_grid[0], cv_grid[1])
    grids = [CV0, CV1]

    prob = np.zeros(CV0.shape)

    for kernel in kernels:
        centers = kernel[:n_cvs]
        weight = float(kernel[n_cvs])
        height = float(kernel[n_cvs + 1])

        gaussian = np.ones(CV0.shape)
        for j in range(n_cvs):
            diff = grids[j] - float(centers[j])
            if periodic[j] is not None:
                period = periodic[j][1] - periodic[j][0]
                diff = diff - period * np.round(diff / period)
            gaussian *= np.exp(-0.5 * (diff / sigma_vals[j]) ** 2)

        prob += weight * height * gaussian

    if sum_weights > 0:
        prob /= sum_weights

    # --- probability → free energy ---
    mask = prob > 0
    fes = np.full(prob.shape, np.nan)
    fes[mask] = -kT * np.log(prob[mask])
    fes -= np.nanmin(fes)
    # Fill unexplored (NaN) regions with the max observed value
    fes_max = np.nanmax(fes)
    fes = np.where(np.isnan(fes), fes_max, fes)

    return fes


def project_fes_1d(cv0_grid, cv1_grid, fes, kT=2.494, project_along='cv0'):
    """
    Project a 2D FES onto 1D by Boltzmann-weighted integration.

    Assumes *fes* has shape ``(n_cv1, n_cv0)`` (xy meshgrid convention).

    Parameters
    ----------
    cv0_grid, cv1_grid : ndarray
    fes : ndarray, shape (n_cv1, n_cv0)
    kT : float
    project_along : 'cv0' | 'cv1'

    Returns
    -------
    grid_1d : ndarray
    fes_1d : ndarray
    """
    prob = np.exp(-fes / kT)

    if project_along == 'cv0':
        # Integrate over cv1 (rows = axis 0) → result shape (n_cv0,)
        prob_1d = np.trapezoid(prob, cv1_grid, axis=0)
        grid_1d = cv0_grid
    elif project_along == 'cv1':
        # Integrate over cv0 (cols = axis 1) → result shape (n_cv1,)
        prob_1d = np.trapezoid(prob, cv0_grid, axis=1)
        grid_1d = cv1_grid
    else:
        raise ValueError("project_along must be 'cv0' or 'cv1'")

    fes_1d = -kT * np.log(np.clip(prob_1d, 1e-300, None))
    fes_1d -= np.min(fes_1d)
    return grid_1d, fes_1d

