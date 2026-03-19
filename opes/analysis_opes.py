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


def _read_kernel_metadata(txt_path):
    """Read barrier/bias-factor metadata from a kernel text file header."""
    meta = {}
    if not os.path.exists(txt_path):
        return meta

    with open(txt_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if line.startswith('# Barrier:'):
                meta['barrier'] = float(line.split(':', 1)[1].split()[0])
            elif line.startswith('# Bias factor:'):
                meta['bias_factor'] = float(line.split(':', 1)[1].split()[0])
            elif line.startswith('# Z_n:'):
                meta['Zn'] = float(line.split(':', 1)[1].split()[0])
            elif line.startswith('# Sum_weights:'):
                meta['sum_weights'] = float(line.split(':', 1)[1].split()[0])
            elif line.startswith('# Kernel_cutoff:'):
                meta['kernel_cutoff'] = float(line.split(':', 1)[1].split()[0])
            elif line.startswith('# Sigma_vals:'):
                meta['sigma_vals'] = [float(x) for x in line.split(':', 1)[1].split()]
    return meta


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
        txt_meta = _read_kernel_metadata(fpath.replace('.npz', '.txt'))

        kernels = data['kernels']
        if len(kernels) == 0:
            continue

        # New-format files store [cv0, cv1, ..., sigma0, sigma1, ..., weight, height]
        # Old-format files store [cv0, cv1, ..., weight, height]
        has_sigma_cols = 'sigma_vals' in data or 'sigma_vals' in txt_meta or kernels.shape[1] > 4
        if has_sigma_cols:
            n_cvs = (kernels.shape[1] - 2) // 2
        else:
            n_cvs = kernels.shape[1] - 2

        # sigma_vals – new-format files store it directly
        if 'sigma_vals' in data:
            sigma_vals = data['sigma_vals'].tolist()
        elif 'sigma_vals' in txt_meta:
            sigma_vals = txt_meta['sigma_vals']
        else:
            # Back-compat: derive from the first height value (equal-sigma)
            height_sample = float(kernels[0, n_cvs + 1])
            sigma_vals = _sigma_from_height(height_sample, n_cvs)

        # sum_weights
        if 'sum_weights' in data:
            sum_weights = float(data['sum_weights'])
        elif 'sum_weights' in txt_meta:
            sum_weights = float(txt_meta['sum_weights'])
        else:
            sum_weights = float(kernels[:, n_cvs].sum())

        # kT / barrier / bias_factor
        if 'kT' in data:
            kT = float(data['kT'])
        else:
            kT = 2.494  # 300 K default

        barrier = float(data['barrier']) if 'barrier' in data else txt_meta.get('barrier', 40.0)
        bias_factor = float(data['bias_factor']) if 'bias_factor' in data else txt_meta.get('bias_factor', 1.0 + barrier / kT)
        kernel_cutoff = float(data['kernel_cutoff']) if 'kernel_cutoff' in data else None

        snapshots.append({
            'step': step,
            'kernels': kernels,
            'sigma_vals': sigma_vals,
            'sum_weights': sum_weights,
            'kT': kT,
            'barrier': barrier,
            'bias_factor': bias_factor,
            'kernel_cutoff': kernel_cutoff,
        })

    return snapshots


def reconstruct_fes_opes(kernels, sigma_vals, sum_weights, kT, cv_grid, periodic=None, barrier=None, bias_factor=None, kernel_cutoff=None):
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

    has_sigma_cols = kernels.shape[1] == 2 * n_cvs + 2
    weight_idx = 2 * n_cvs if has_sigma_cols else n_cvs
    height_idx = weight_idx + 1

    if kernel_cutoff is None:
        if barrier is None:
            barrier = 40.0
        if bias_factor is None:
            bias_factor = 1.0 + barrier / kT
        bias_prefactor = 1.0 - 1.0 / bias_factor
        kernel_cutoff = math.sqrt(2.0 * barrier / (bias_prefactor * kT))
    kernel_cutoff2 = kernel_cutoff ** 2
    val_at_cutoff = math.exp(-0.5 * kernel_cutoff2)

    # xy meshgrid: shape (n_cv1, n_cv0)
    CV0, CV1 = np.meshgrid(cv_grid[0], cv_grid[1])
    grids = [CV0, CV1]

    prob = np.zeros(CV0.shape)

    for kernel in kernels:
        centers = kernel[:n_cvs]
        weight = float(kernel[weight_idx])
        height = float(kernel[height_idx])
        kernel_sigmas = [float(kernel[n_cvs + j]) for j in range(n_cvs)] if has_sigma_cols else list(sigma_vals)

        norm2 = np.zeros(CV0.shape)
        for j in range(n_cvs):
            diff = grids[j] - float(centers[j])
            if periodic[j] is not None:
                period = periodic[j][1] - periodic[j][0]
                diff = diff - period * np.round(diff / period)
            norm2 += (diff / kernel_sigmas[j]) ** 2

        gaussian = np.where(norm2 < kernel_cutoff2, np.exp(-0.5 * norm2) - val_at_cutoff, 0.0)

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

