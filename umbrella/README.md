# Umbrella sampling

Fixed harmonic windows in OpenMM, with binless MBAR reweighting using PyMBAR.
The bias is `0.5 * sum(k_i * displacement_i**2)`. Periodic CVs use the shortest
displacement around their circle, including across the `-pi/pi` seam. Window
centers and spring constants are global parameters and can be updated in an
existing Context.

## Run alanine dipeptide

From the repository root:

```bash
source /home/dom/anaconda3/etc/profile.d/conda.sh
conda activate openmm
python -m pip install -r umbrella/requirements.txt
export OPENMM_CPU_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
python umbrella/run_aladipep.py --dimension 1 --workers 6
python umbrella/plot_fes_umbrella.py umbrella/output_1d
python umbrella/run_aladipep.py --dimension 2 --workers 6
python umbrella/plot_fes_umbrella.py umbrella/output_2d
python -m unittest discover -s umbrella -p 'test_*.py' -v
```

The example uses the same vacuum Amber14 system, torsion atom indices and 300 K
temperature as the metaD/OPES examples. H-bond constraints, a 2 fs timestep and
the Langevin middle integrator (1/ps friction) are used. Each CPU worker has one
OpenMM thread. Windows have independent integrators/seeds and start from the
repository PDB, rather than inheriting a previous window's trajectory.

| Setting | 1D | 2D |
| --- | --- | --- |
| Biased CVs | phi | phi, psi |
| Windows | 24 | 16 x 16 |
| Angular spacing | 15 degrees | 22.5 degrees |
| Spring constants | 100 kJ/mol/rad² | 100 kJ/mol/rad² on each CV |
| Preparation | minimize, then 5 ps with k=1000 | same |
| Equilibration under production bias | 50 ps/window | 50 ps/window |
| Production | 500 ps/window | 250 ps/window |
| Save interval | 0.2 ps | 0.2 ps |
| Aggregate production | 12 ns | 64 ns |

Preparation and equilibration samples are excluded from reconstruction. At
300 K, k=100 corresponds to a harmonic thermal width of about 0.158 rad
(9 degrees). Actual distributions also depend on the physical potential;
spacing alone does not establish adequate overlap. These defaults are test
protocols, not a claim of converged thermodynamics. Use `--help` to change the
window grid, spring constant, sampling duration or output directory.

If strong-restraint preparation produces NaN coordinates, the runner retries
from a fresh Context: minimize near the initial CVs, then gently steer centers
to the target using the production spring constant and a 0.2 fs timestep for
the same 5 ps preparation duration. Equilibration and production still use
2 fs and the fixed target restraint. The actual preparation method and timestep
are recorded in the window file. A second failure propagates; queued work is
cancelled and completed windows are retained.

Each window writes a production-only `window_XXXX.npz`. A manifest records the
protocol and complete grid. Rerunning the identical command skips completed
windows; changing the protocol requires a new output directory. Files are
written with atomic replacement. Reconstruction checks for missing windows
when a manifest is present.

## Module API

```python
import numpy as np
from openmm import CustomTorsionForce
from umbrella import UmbrellaSampling, UmbrellaVariable

phi = CustomTorsionForce('theta')
phi.addTorsion(4, 6, 8, 14)
bias = UmbrellaSampling(system, [
    UmbrellaVariable(phi, center=-1.2, force_constant=100.,
                     periodic=(-np.pi, np.pi))
])
# Construct Simulation after adding the umbrella force.
bias.set_window(simulation.context, centers=[-0.9])
bias.step(simulation, 1000)
values = bias.getCollectiveVariables(simulation)
centers, spring_constants = bias.get_window(simulation.context)
```

Supply numeric CV values in the native units returned by the CV Force:
radians for a torsion and nanometers for a distance. Spring constants are
numeric kJ/mol per squared native CV unit. `periodic=None` gives a nonperiodic
harmonic restraint. Each CV Force must be fresh and transferred to the umbrella
force before creating a Context. Use distinct `prefix` values for multiple
biases in one System. The bias uses an otherwise unused force group.

`UmbrellaCVReporter` supplies a text CV reporter for other simulations; close
it explicitly or use it as a context manager. The parallel example writes
compact NumPy window files directly instead.
The reporter also records the actual window centers and springs on every row.
`variables` contains constructor defaults; use `get_window(context)` when
recording a window changed through global parameters.

## Analysis and diagnostics

`WindowData(samples, center, force_constants)` and `reconstruct(...)` provide
the analysis API. Histogram reconstruction supports one or two CVs. Periodic
CVs use their periodic bounds; nonperiodic CVs require the `bounds` argument
and reject samples outside the histogram range. All windows must have the same
temperature and underlying physical Hamiltonian. The common physical energy
cancels from the MBAR equations, so evaluating all umbrella energies at every
sample suffices. Unequal sample counts are handled through `N_k`.

Samples are decorrelated separately in each window using the maximum estimated
statistical inefficiency of the umbrella energy and each CV's sine/cosine.
PyMBAR computes unbiased weights through an extra unsampled zero-bias state.
Weighted histogram probabilities yield the FES; empty bins remain infinite
and are masked in plots. There is no interpolation or smoothing. 2D arrays
are indexed `[phi, psi]`; plotting transposes them. The 1D projection sums
probabilities over psi, rather than taking a minimum over psi.

Outputs include `fes.npz`, `metrics.json`, `umbrella-results.png` and a PDF.
The plots show:

- 2D FES, or 1D biased-window distributions.
- The unbiased 1D phi profile at successive production durations.
- DeltaF versus production time **per window**, with aggregate production on
  a secondary axis. Preparation and equilibration are not counted on this axis.
- Decorrelated sample counts, unbiased effective sample size, and the MBAR
  overlap spectral gap. A vanishing gap rejects reconstruction.

For continuity with the existing examples, the projected-minima difference is
`min(F(phi in [-pi/2,-pi/4))) - min(F(phi in [0,3pi/4)))`. This is not a
basin population free energy. A second curve reports `-kT log(P_A/P_B)` for
the same phi intervals. These intervals are comparison regions, not a
complete structural classification of alanine conformers. Grid centers define
region membership; finite bin size differs slightly from the old 200-point
inclusive grids and index slices.

Metrics also compare the first and second production halves. Connected
windows, stable DeltaF, and many decorrelated samples are useful diagnostics,
but cannot exclude slow hidden degrees of freedom or preparation bias.
Especially in phi-only sampling, psi may fail to equilibrate. No confidence
intervals are claimed; use independent repeats, longer equilibration and
sampling, or block/replicate uncertainty estimates for quantitative work.

## Protocols consulted

- [PLUMED Masterclass 21.3](https://github.com/plumed/masterclass-21-3/blob/main/INSTRUCTIONS.md)
  demonstrates vacuum alanine dipeptide with 32 uniformly spaced phi windows,
  k=200 kJ/mol/rad², and 200,000 steps per window. It emphasizes overlap,
  discarding initial transients in production work, WHAM reweighting and
  bootstrap population errors. Its periodic grid excludes the duplicated
  endpoint; this example follows that rule.
- [Levy group 2D umbrella example](https://ronlevygroup.cst.temple.edu/software/UWHAM_and_SWHAM_webpage/02_AlaD_Umbrella_Sampling/readme.html)
  uses an implicit-solvent Amber99SB system with 24 x 24 windows, 15-degree
  spacing and k=200 kJ/mol/rad². Our coarser 16 x 16 test uses a weaker spring
  to retain a similar spacing-to-thermal-width ratio. The different solvent
  and force field mean its FES is not a reference for our vacuum system.
- [RSC Advances, DOI 10.1039/D2RA08180F](https://pubs.rsc.org/en/content/articlehtml/2023/ra/d2ra08180f)
  illustrates a denser 36 x 36 alanine grid at 10-degree intervals, 500 ps
  per window and k=100 kcal/mol/rad², analyzed with 2D WHAM. This is another
  solvent-model study, not a transferable FES benchmark.
- [PyMBAR documentation](https://pymbar.readthedocs.io/en/stable/mbar.html)
  specifies decorrelation, reduced-potential inputs, unsampled states,
  effective sample counts and overlap diagnostics used here.
