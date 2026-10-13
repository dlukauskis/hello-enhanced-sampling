"""Reconstruct FES and equal-per-window-time convergence from saved windows."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if __package__:
    from .analysis_umbrella import load_windows, reconstruct
else:
    from analysis_umbrella import load_windows, reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.1, .25, .5, .75, 1.])
    args = parser.parse_args()
    fractions = sorted(set(args.fractions + [1.]))
    if any(not 0 < x <= 1 for x in fractions):
        parser.error("Fractions must be in (0, 1]")
    results, metrics = [], []
    for fraction in fractions:
        windows, metadata = load_windows(args.directory, fraction)
        result = reconstruct(windows, metadata["periodic"], metadata["temperature"], args.bins)
        results.append(result)
        elapsed_ps = len(windows[0].samples) * metadata["stride_ps"]
        metrics.append(dict(fraction=fraction, production_ps_per_window=elapsed_ps,
                            aggregate_production_ns=elapsed_ps * len(windows) / 1000,
                            delta_min_kj_mol=result["delta_min"], delta_population_kj_mol=result["delta_population"],
                            effective_samples=result["effective_samples"], overlap_gap=result["overlap_gap"],
                            retained_samples=int(sum(result["counts"])),
                            min_window_samples=int(min(result["counts"])),
                            empty_bin_fraction=float(np.mean(~np.isfinite(result["fes"])))))
        if min(result["counts"]) < 20:
            print("Sampling warning: at least one window has <20 decorrelated samples", flush=True)
        print(json.dumps(metrics[-1]), flush=True)
    final = results[-1]
    # Temporal halves expose drift that a full-trajectory average can conceal.
    halves = []
    for start, stop in [(0., .5), (.5, 1.)]:
        windows, _ = load_windows(args.directory, stop, start)
        r = reconstruct(windows, metadata["periodic"], metadata["temperature"], args.bins)
        halves.append(dict(start_fraction=start, stop_fraction=stop,
                           delta_min_kj_mol=r["delta_min"], delta_population_kj_mol=r["delta_population"],
                           overlap_gap=r["overlap_gap"]))
    diagnostics = dict(convergence=metrics, temporal_halves=halves,
                       counts=final["counts"].tolist(), raw_counts=final["raw_counts"].tolist(),
                       inefficiencies=final["inefficiencies"].tolist(), metadata=metadata,
                       delta_definition="min F(phi in [-pi/2,-pi/4)) - min F(phi in [0,3pi/4))",
                       population_definition="-kT ln(P[-pi/2,-pi/4)/P[0,3pi/4))",
                       uncertainty="No confidence intervals; temporal halves and overlap are diagnostics, not proof of convergence")
    (args.directory / "metrics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    np.savez_compressed(args.directory / "fes.npz", fes=final["fes"], phi=final["grid"][0],
                        psi=final["grid"][1] if len(final["grid"]) == 2 else np.array([]),
                        profile=final["profile"], probability=final["probability"],
                        overlap=final["overlap_matrix"], fractions=fractions,
                        profiles=np.array([r["profile"] for r in results]),
                        delta_min=np.array([r["delta_min"] for r in results]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    if len(final["grid"]) == 2:
        image = axes[0, 0].pcolormesh(final["grid"][0], final["grid"][1],
                                     np.ma.masked_invalid(final["fes"].T),
                                     shading="nearest", cmap="viridis", vmin=0, vmax=35)
        fig.colorbar(image, ax=axes[0, 0], label="F (kJ/mol)", extend="max")
        axes[0, 0].set_facecolor("0.9")
        axes[0, 0].set(xlabel="φ (rad)", ylabel="ψ (rad)", title="2D unbiased FES (empty bins masked)")
    else:
        windows, _ = load_windows(args.directory)
        for w in windows:
            axes[0, 0].hist(w.samples[:, 0], bins=72, range=(-np.pi, np.pi),
                            density=True, histtype="step", alpha=.5)
        axes[0, 0].set(xlabel="φ (rad)", ylabel="Biased density", title="Window distributions")
    for metric, result in zip(metrics, results):
        axes[0, 1].plot(result["grid"][0], np.ma.masked_invalid(result["profile"]),
                        label=f"{metric['production_ps_per_window']:g} ps/window")
    axes[0, 1].set(xlabel="φ (rad)", ylabel="F (kJ/mol)", title="1D φ FES convergence", ylim=(0, 35))
    axes[0, 1].legend(fontsize=8)
    times = [m["production_ps_per_window"] for m in metrics]
    axes[1, 0].plot(times, [m["delta_min_kj_mol"] for m in metrics], "o-", label="Projected minima (repo convention)")
    axes[1, 0].plot(times, [m["delta_population_kj_mol"] for m in metrics], "s-", label="Integrated φ regions")
    axes[1, 0].set(xlabel="Production time per window (ps)", ylabel="ΔF (kJ/mol)", title="ΔF versus accumulated sampling")
    axes[1, 0].legend(fontsize=8)
    secondary = axes[1, 0].secondary_xaxis("top", functions=(lambda x: x * len(windows) / 1000,
                                                                        lambda x: x * 1000 / len(windows)))
    secondary.set_xlabel("Aggregate production (ns)", fontsize=9)
    axes[1, 1].plot(final["counts"], label="Decorrelated samples")
    axes[1, 1].set(xlabel="Window index", ylabel="Retained samples", title=f"Overlap gap={final['overlap_gap']:.3g}; unbiased ESS={final['effective_samples']:.0f}")
    fig.suptitle(f"Vacuum alanine dipeptide: {metadata['dimension']}D umbrella sampling, {metadata['temperature']:g} K")
    fig.savefig(args.directory / "umbrella-results.png", dpi=180)
    fig.savefig(args.directory / "umbrella-results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
