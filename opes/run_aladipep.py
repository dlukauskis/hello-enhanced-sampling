from sys import stdout

from openmm import *
from openmm.app import *
import openmm.unit as unit
import opes
import numpy as np
import os
from opes_reporter import OPESCVReporter
from plot_fes_opes import main as plot_fes_opes

total_steps = 2500000  # 5 ns with 2 fs timestep, comparable to wt-metad reference
# total_steps = 500000  # 1 ns with 2 fs timestep, enough for 1-2 recrossings in wt-metad
# total_steps = 5000  # 0.01 ns with 2 fs timestep, just a smoke test
kernel_pace = 500
stride = 500


# Create a System for alanine dipeptide in vacuo
pdb_file = PDBFile('../benchmark_systems/aladipep/system.pdb')
forcefield = ForceField('amber14-all.xml')

system = forcefield.createSystem(
    pdb_file.topology,
    nonbondedMethod=NoCutoff,
    constraints=HBonds,
)

# Define collective variables for phi and psi.
cv1 = CustomTorsionForce('theta')
cv1.addTorsion(4, 6, 8, 14)
cv2 = CustomTorsionForce('theta')
cv2.addTorsion(6, 8, 14, 16)

# Create OPES bias
opes_bias = opes.OPES(
    system=system,
    variables=[cv1, cv2],
    temperature=300 * unit.kelvin,
    barrier=40 * unit.kilojoules_per_mole,
    sigma=[0.35 * unit.radian, 0.35 * unit.radian],
    stride=kernel_pace,
    saveFrequency=50000,  # every 100 depositions (100 ps); was stride=500 → every deposition!
    biasDir='output',
    periodic=[(-np.pi, np.pi), (-np.pi, np.pi)],  # Both phi and psi are periodic!
)

# Run simulation
integrator = LangevinIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond,
)
simulation = Simulation(pdb_file.topology, system, integrator)
simulation.context.setPositions(pdb_file.positions)

# add CV reporter
os.makedirs(opes_bias.biasDir, exist_ok=True)
cv_report_file = os.path.join(opes_bias.biasDir, 'cv_history.txt')
cv_reporter = OPESCVReporter(cv_report_file, kernel_pace, opes_bias)
simulation.reporters.append(cv_reporter)

simulation.reporters.append(StateDataReporter(
    stdout, 50000, step=True,
    temperature=True,progress=True,
    remainingTime=True,speed=True,
    totalSteps=total_steps, separator=' ')
)

# Run with OPES
opes_bias.step(simulation, total_steps)

# Extract free energy
cv_grid = [np.linspace(-3.14, 3.14, 200), np.linspace(-3.14, 3.14, 200)]
fes = opes_bias.getFreeEnergy(cv_grid)

# save fes for later analysis
fes_out_fpath = os.path.join('output', 'opes_output_fes.npz')
np.savez_compressed(fes_out_fpath, fes=fes, grid0=cv_grid[0], grid1=cv_grid[1])

# save kernels and cv history
if opes_bias.biasDir:
    opes_bias.saveKernels()

# Show the same 4-panel results figure as the wt-metaD example:
#   (1) CV evolution, (2) 2D FES, (3) 1D FES convergence, (4) ΔF over time
plot_fes_opes(
    output_dir=opes_bias.biasDir,
    fig_fname=os.path.join(opes_bias.biasDir, 'opes-aladipep-results.png'),
    periodic=[(-np.pi, np.pi), (-np.pi, np.pi)],
)
