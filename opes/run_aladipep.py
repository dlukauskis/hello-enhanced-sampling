from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond, radian
import opes
import numpy as np
import os

total_steps = 2500000  # 5 ns with 2 fs timestep, enough for 5-10 crossings
# total_steps = 500000  # 1 ns with 2 fs timestep, enough for 1-2 crossings
hills_pace = 500
hills_write_pace = 500


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
    temperature=300 * kelvin,
    barrier=10 * kilojoules_per_mole,
    sigma=[0.35 * radian, 0.35 * radian],
    stride=hills_pace,
    biasDir='output',
    periodic=[(-np.pi, np.pi), (-np.pi, np.pi)],  # Both phi and psi are periodic!
    initial_height=1.2,  # kJ/mol
)

# Run simulation
integrator = LangevinIntegrator(
    300 * kelvin, 1.0 / picosecond, 0.002 * picosecond,
)
simulation = Simulation(pdb_file.topology, system, integrator)
simulation.context.setPositions(pdb_file.positions)

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
fes_out_fpath = os.path.join('results', 'opes_output_fes.npz')
np.savez_compressed(fes_out_fpath, fes=fes, grid0=cv_grid[0], grid1=cv_grid[1])
