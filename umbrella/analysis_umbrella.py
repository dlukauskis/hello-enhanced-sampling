"""Binless MBAR reweighting and 1D/2D histogram FES reconstruction.

All windows must share the underlying Hamiltonian and temperature. The common
physical energy cancels in MBAR, so only harmonic umbrella energies are needed.
Input trajectories must already exclude preparation and equilibration.
"""
from dataclasses import dataclass
import numpy as np
from openmm import unit
from umbrella import bias_energy


@dataclass
class WindowData:
    samples: np.ndarray
    center: np.ndarray
    force_constants: np.ndarray


def thermal_energy(temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite (kelvin)")
    return (unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin).value_in_unit(
        unit.kilojoules_per_mole)


def independent_indices(samples, window, periodic):
    """Conservatively decorrelate using each CV and the bias energy.

    Circular CVs use sine and cosine, avoiding artificial seam jumps. This
    diagnoses measured observables, not equilibration of hidden coordinates.
    """
    from pymbar import timeseries
    observables = [bias_energy(samples, window.center, window.force_constants, periodic)]
    for i, bounds in enumerate(periodic):
        if bounds is None:
            observables.append(samples[:, i])
        else:
            angle = 2 * np.pi * (samples[:, i] - bounds[0]) / (bounds[1] - bounds[0])
            observables.extend((np.sin(angle), np.cos(angle)))
    gs = [timeseries.statistical_inefficiency(x) for x in observables
          if np.var(x) > 1e-14]
    if not gs:
        raise ValueError("Window has no measured fluctuations; cannot estimate independent samples")
    g = max([1.0] + gs)
    indices = np.asarray(timeseries.subsample_correlated_data(
        np.arange(len(samples)), g=g), dtype=int)
    return indices, g


def reconstruct(windows, periodic, temperature=300., bins=72, decorrelate=True, bounds=None):
    """Return unbiased probabilities, FES and overlap diagnostics.

    Histogram axes follow CV order (phi, psi), not plotting image order.
    Empty bins have infinite free energy. No smoothing or gap filling is done.
    Nonperiodic CVs use ``None`` in periodic and require histogram ``bounds``.
    All samples must lie in the supplied bounds, preventing silent truncation.
    """
    from pymbar import MBAR
    kt = thermal_energy(temperature)
    dimension = len(periodic)
    if dimension not in (1, 2) or not windows:
        raise ValueError("Supply windows for one or two CVs")
    ranges = periodic if bounds is None else bounds
    if len(ranges) != dimension:
        raise ValueError("Supply histogram bounds for each CV")
    for limits in ranges:
        if (limits is None or len(limits) != 2 or not np.isfinite(limits).all() or
                limits[1] <= limits[0]):
            raise ValueError("Bounds must be finite and increasing")
    for limits in periodic:
        if limits is not None and (len(limits) != 2 or not np.isfinite(limits).all() or limits[1] <= limits[0]):
            raise ValueError("Periodic bounds must be finite and increasing")
    samples, counts, gs, raw_counts = [], [], [], []
    for w in windows:
        x = np.asarray(w.samples, dtype=float)
        c, k = np.asarray(w.center), np.asarray(w.force_constants)
        if (x.ndim != 2 or x.shape[1] != dimension or len(x) < 20 or
                not np.isfinite(x).all() or c.shape != (dimension,) or
                k.shape != (dimension,) or not np.isfinite(c).all() or
                not np.isfinite(k).all() or (k <= 0).any()):
            raise ValueError("Invalid window: require >=20 finite samples and positive springs")
        x = x.copy()
        for i, limits in enumerate(periodic):
            if limits is not None:
                low, high = limits
                x[:, i] = low + (x[:, i] - low) % (high - low)
            low, high = ranges[i]
            if (x[:, i] < low).any() or (x[:, i] > high).any():
                raise ValueError("Histogram bounds exclude samples")
        ids, g = independent_indices(x, w, periodic) if decorrelate else (np.arange(len(x)), 1.)
        samples.append(x[ids])
        counts.append(len(ids))
        raw_counts.append(len(x))
        gs.append(g)
    x = np.concatenate(samples)
    # Extra unsampled state has zero bias, giving normalized unbiased weights.
    u = np.empty((len(windows) + 1, len(x)))
    for i, w in enumerate(windows):
        u[i] = bias_energy(x, w.center, w.force_constants, periodic) / kt
    u[-1] = 0
    mbar = MBAR(u, np.r_[counts, 0], solver_protocol="robust", relative_tolerance=1e-9)
    weights = mbar.W_nk[:, -1]
    overlap = mbar.compute_overlap()
    gap = float(overlap["scalar"])
    if gap < 1e-8:
        raise ValueError("Window overlap is disconnected or numerically negligible; add sampling/windows")
    if isinstance(bins, int):
        bins = [bins] * dimension
    if len(bins) != dimension or any(int(n) != n or n < 2 for n in bins):
        raise ValueError("Provide at least two bins per CV")
    edges = [np.linspace(low, high, int(n) + 1) for (low, high), n in zip(ranges, bins)]
    probability, _ = np.histogramdd(x, bins=edges, weights=weights)
    volume = float(np.prod([e[1] - e[0] for e in edges]))
    with np.errstate(divide="ignore"):
        fes = -kt * np.log(probability / volume)
    fes -= np.min(fes)
    phi_probability = probability if dimension == 1 else probability.sum(axis=1)
    with np.errstate(divide="ignore"):
        profile = -kt * np.log(phi_probability / (edges[0][1] - edges[0][0]))
    profile -= np.min(profile)
    grid = [(e[1:] + e[:-1]) * .5 for e in edges]
    # Repository comparison convention: difference between projected minima.
    a = (grid[0] >= -np.pi / 2) & (grid[0] < -np.pi / 4)
    b = (grid[0] >= 0) & (grid[0] < 3 * np.pi / 4)
    if (periodic[0] is not None and np.allclose(periodic[0], [-np.pi, np.pi])
            and a.any() and b.any()):
        delta_min = float(np.min(profile[a]) - np.min(profile[b]))
        pa, pb = phi_probability[a].sum(), phi_probability[b].sum()
        delta_population = float(-kt * np.log(pa / pb)) if pa > 0 and pb > 0 else float("nan")
    else:
        delta_min = delta_population = float("nan")
    return dict(fes=fes, profile=profile, probability=probability, grid=grid,
                delta_min=delta_min, delta_population=delta_population,
                effective_samples=float(1 / np.sum(weights**2)), overlap_gap=gap,
                overlap_matrix=overlap["matrix"], counts=np.array(counts),
                raw_counts=np.array(raw_counts), inefficiencies=np.array(gs),
                window_free_energies=mbar.f_k[:-1] * kt)


def load_windows(directory, fraction=1., start_fraction=0.):
    """Read production-only window files; slice all windows at equal elapsed time."""
    from pathlib import Path
    if not 0 <= start_fraction < fraction <= 1:
        raise ValueError("Require 0 <= start_fraction < fraction <= 1")
    files = sorted((file for file in Path(directory).glob("window_*.npz")
                    if file.stem.removeprefix("window_").isdigit()),
                   key=lambda file: int(file.stem.removeprefix("window_")))
    if not files:
        raise ValueError("No umbrella window files found")
    windows = []
    metadata = None
    sample_count = None
    preparation_methods, preparation_timesteps = [], []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            current = {key: data[key].tolist() for key in (
                "temperature", "periodic", "production_ps", "equilibration_ps", "stride_ps", "dimension")}
            if metadata is not None and current != metadata:
                raise ValueError("Window metadata differ: Hamiltonian/temperature/protocol must match")
            metadata = current
            n = len(data["samples"])
            if sample_count is not None and n != sample_count:
                raise ValueError("Equal-time convergence requires equal trajectory lengths")
            sample_count = n
            expected_samples = round(current["production_ps"] / current["stride_ps"])
            if n != expected_samples:
                raise ValueError("Trajectory length differs from saved production duration")
            windows.append(WindowData(data["samples"][int(n * start_fraction):int(n * fraction)],
                                      data["center"].copy(), data["force_constants"].copy()))
            preparation_methods.append(str(data["preparation_method"].item())
                                       if "preparation_method" in data else "strong")
            preparation_timesteps.append(str(float(data["preparation_timestep_ps"]))
                                         if "preparation_timestep_ps" in data else "0.002")
    # A manifest prevents silent reconstruction from an incomplete grid.
    import json
    manifest_path = Path(directory) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = np.asarray(manifest["centers"])
        actual = np.asarray([w.center for w in windows])
        if expected.shape != actual.shape or not np.allclose(expected, actual, atol=1e-12, rtol=0):
            raise ValueError("Window grid is incomplete or differs from manifest")
    from collections import Counter
    metadata = dict(metadata, preparation_method_counts=dict(Counter(preparation_methods)),
                    preparation_timestep_counts_ps=dict(Counter(preparation_timesteps)))
    return windows, metadata
