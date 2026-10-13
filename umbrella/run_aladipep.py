"""Independent umbrella simulations of vacuum alanine dipeptide.

Example: python umbrella/run_aladipep.py --dimension 2 --workers 6
Run from any directory. CPU workers each use one OpenMM thread.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

# Must precede numpy/OpenMM imports in spawned workers.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PreparationError(RuntimeError):
    """A window failed before equilibration/production."""


def run_window(task):
    index, center, config = task
    try:
        return _run_window(task)
    except PreparationError:
        # Some high-energy targets distort a structure during stiff restrained
        # minimization. Retry from a fresh Context by gradually moving a weaker
        # restraint, rather than retaining any part of a failed trajectory.
        if config.get("preparation_method", "strong") != "strong":
            raise
        print(f"Window {index}: retrying with gentle steering at 0.2 fs", flush=True)
        return _run_window((index, center, dict(config, preparation_substeps=10,
                                               preparation_method="steered")))


def _run_window(task):
    from openmm import CustomTorsionForce, LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, PDBFile, Simulation, NoCutoff, HBonds
    from umbrella import UmbrellaSampling, UmbrellaVariable, displacement
    index, center, config = task
    target = Path(config["output"]) / f"window_{index:04d}.npz"
    if target.exists():
        # Exact manifest compatibility is checked before starting the pool.
        with np.load(target, allow_pickle=False) as data:
            if (len(data["samples"]) != config["production_steps"] // config["stride_steps"] or
                    data["samples"].shape[1:] != (config["dimension"],) or
                    not np.array_equal(data["center"], center) or
                    not np.array_equal(data["force_constants"], np.full(config["dimension"], config["spring"])) or
                    float(data["temperature"]) != config["temperature"] or
                    int(data["seed"]) != config["seed"] + index or
                    not np.isfinite(data["samples"]).all()):
                raise ValueError(f"Invalid existing window: {target}")
        return index, "existing"
    started = time.perf_counter()
    pdb = PDBFile(str(ROOT / "benchmark_systems/aladipep/system.pdb"))
    system = ForceField("amber14-all.xml").createSystem(
        pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    variables = []
    for i, atoms in enumerate(((4, 6, 8, 14), (6, 8, 14, 16))[:config["dimension"]]):
        cv = CustomTorsionForce("theta")
        cv.addTorsion(*atoms)
        variables.append(UmbrellaVariable(cv, center[i], config["spring"], (-np.pi, np.pi)))
    bias = UmbrellaSampling(system, variables)
    seed = config["seed"] + index
    substeps = config.get("preparation_substeps", 1)
    integrator = LangevinMiddleIntegrator(config["temperature"] * unit.kelvin,
                                         1 / unit.picosecond, .002 / substeps * unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    simulation = Simulation(pdb.topology, system, integrator,
                            Platform.getPlatformByName("CPU"), {"Threads": "1"})
    simulation.context.setPositions(pdb.positions)
    steered = config.get("preparation_method", "strong") == "steered"
    if steered:
        initial = np.array(bias.getCollectiveVariables(simulation))
        bias.set_window(simulation.context, initial)
    else:
        bias.set_window(simulation.context, center, [1000.] * config["dimension"])
    try:
        simulation.minimizeEnergy(maxIterations=1000)
        simulation.context.setVelocitiesToTemperature(config["temperature"] * unit.kelvin, seed)
        if steered:
            initial = np.array(bias.getCollectiveVariables(simulation))
            delta = displacement(center, initial, [(-np.pi, np.pi)] * config["dimension"])
            total = config["preparation_steps"] * substeps
            segments = min(100, max(1, total))
            length, remainder = divmod(total, segments)
            for segment in range(segments):
                bias.set_window(simulation.context, initial + delta * (segment + 1) / segments)
                simulation.step(length + (segment < remainder))
        else:
            simulation.step(config["preparation_steps"] * substeps)
    except Exception as error:
        from openmm import OpenMMException
        if isinstance(error, OpenMMException) and "NaN" in str(error):
            raise PreparationError(f"Window {index} preparation produced NaN coordinates") from error
        raise
    integrator.setStepSize(.002 * unit.picoseconds)
    bias.set_window(simulation.context, center, [config["spring"]] * config["dimension"])
    simulation.step(config["equilibration_steps"])
    samples = np.empty((config["production_steps"] // config["stride_steps"], config["dimension"]))
    for i in range(len(samples)):
        simulation.step(config["stride_steps"])
        samples[i] = bias.getCollectiveVariables(simulation)
    if not np.isfinite(samples).all():
        raise RuntimeError(f"Nonfinite CV in window {index}")
    # Rename atomically so interrupted jobs cannot leave apparently complete files.
    temporary = target.with_suffix(".tmp.npz")
    actual_center, actual_springs = bias.get_window(simulation.context)
    np.savez_compressed(temporary, samples=samples, center=actual_center,
                        force_constants=actual_springs,
                        temperature=config["temperature"], seed=seed,
                        periodic=np.tile([-np.pi, np.pi], (config["dimension"], 1)),
                        production_ps=config["production_steps"] * .002,
                        equilibration_ps=config["equilibration_steps"] * .002,
                        preparation_timestep_ps=.002 / substeps,
                        preparation_method="steered" if steered else "strong",
                        stride_ps=config["stride_steps"] * .002,
                        dimension=config["dimension"], wall_seconds=time.perf_counter() - started)
    temporary.replace(target)
    return index, f"{time.perf_counter() - started:.1f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=2)
    parser.add_argument("--windows", type=int, help="Windows per axis (default: 24 in 1D, 16 in 2D)")
    parser.add_argument("--spring", type=float, default=100., help="kJ/mol/radian^2")
    parser.add_argument("--production-ps", type=float, help="Default 500 ps (1D) or 250 ps (2D)")
    parser.add_argument("--equilibration-ps", type=float, default=50.)
    parser.add_argument("--preparation-ps", type=float, default=5.)
    parser.add_argument("--stride-ps", type=float, default=.2)
    parser.add_argument("--temperature", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    count = args.windows if args.windows is not None else (24 if args.dimension == 1 else 16)
    production = args.production_ps if args.production_ps is not None else (500. if args.dimension == 1 else 250.)
    if (count < 2 or args.workers < 1 or args.seed < 1 or
            not all(np.isfinite(x) and x > 0 for x in (args.spring, production, args.stride_ps, args.temperature)) or
            not all(np.isfinite(x) and x >= 0 for x in (args.equilibration_ps, args.preparation_ps))):
        parser.error("Invalid grid, timing, seed, temperature or spring constant")
    def steps(ps):
        n = round(ps / .002)
        if not np.isclose(n * .002, ps, rtol=0, atol=1e-9):
            parser.error("Times must be multiples of the 2 fs timestep")
        return n
    config = dict(dimension=args.dimension, spring=args.spring, temperature=args.temperature,
                  seed=args.seed, production_steps=steps(production),
                  equilibration_steps=steps(args.equilibration_ps),
                  preparation_steps=steps(args.preparation_ps), stride_steps=steps(args.stride_ps),
                  output=str((args.output or ROOT / "umbrella" / f"output_{args.dimension}d").resolve()),
                  forcefield="amber14-all.xml", timestep_ps=.002,
                  friction_per_ps=1., integrator="LangevinMiddleIntegrator", system="vacuum alanine dipeptide")
    if (config["stride_steps"] < 1 or config["production_steps"] % config["stride_steps"] or
            config["production_steps"] < 20 * config["stride_steps"]):
        parser.error("Production must be a multiple of the sampling stride and provide >=20 samples")
    if args.seed + count**args.dimension - 1 > 2147483647:
        parser.error("Window seeds must fit OpenMM's signed 32-bit seed range")
    angles = -np.pi + (np.arange(count) + .5) * 2 * np.pi / count
    centers = np.array(np.meshgrid(*([angles] * args.dimension), indexing="ij")).reshape(args.dimension, -1).T
    manifest = dict(config, centers=centers.tolist(), windows_per_axis=count)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing output has a different protocol; choose a new directory")
    elif list(output.glob("window_*.npz")):
        raise ValueError("Existing windows lack a manifest; choose a new directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    tasks = [(i, center, config) for i, center in enumerate(centers)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_window, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                index, duration = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                print("Window failed; cancelling queued tasks. Completed windows are preserved.", flush=True)
                raise
            print(f"{done}/{len(tasks)} window {index}: {duration}", flush=True)
    print(f"Completed in {time.perf_counter() - started:.1f}s; production aggregate {len(tasks) * production / 1000:g} ns", flush=True)


if __name__ == "__main__":
    main()
