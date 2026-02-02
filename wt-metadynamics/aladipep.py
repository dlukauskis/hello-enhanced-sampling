from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond
from metadynamics import Metadynamics, BiasVariable
from metadynamicsreporter import MetadynamicsReporter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create a System for alanine dipeptide in water.
total_steps = 100000

pdb_file = PDBFile('../benchmark_systems/aladipep/system.pdb')
forcefield = ForceField('amber14-all.xml')

system = forcefield.createSystem(
    pdb_file.topology,
    nonbondedMethod=NoCutoff,
    constraints=HBonds
)

# Define collective variables for phi and psi.
cv1 = CustomTorsionForce('theta')
cv1.addTorsion(4, 6, 8, 14)
sigma_cv0 = 0.35
phi = BiasVariable(cv1, -np.pi, np.pi, sigma_cv0, True)
cv2 = CustomTorsionForce('theta')
cv2.addTorsion(6, 8, 14, 16)
sigma_cv1 = 0.35
psi = BiasVariable(cv2, -np.pi, np.pi, sigma_cv1, True)

# Set up the simulation.
bias_f = 6
meta = Metadynamics(
    system, [phi, psi], 300.0 * kelvin,
    bias_f, 1.2 * kilojoules_per_mole, 100
)
integrator = LangevinIntegrator(
    300 * kelvin, 1.0 / picosecond, 0.002 * picosecond
)
simulation = Simulation(pdb_file.topology, system, integrator)
simulation.context.setPositions(pdb_file.positions)

simulation.reporters.append(StateDataReporter(
    stdout, 50000, step=True,
    temperature=True,progress=True,
    remainingTime=True,speed=True,
    totalSteps=total_steps, separator=' ')
)

meta.reporters.append(
    MetadynamicsReporter('HILLS', 1000)
)

meta.step(simulation, total_steps)

# Plot CV time series
df_cv = pd.read_csv('HILLS', delimiter='\s+', skiprows=7, header=None)
df_cv.columns = ['step', 'time(ps)', 'cv0', 'cv1', 'sigma_cv0', 'sigma_cv1', 'hill_height', 'bias_factor']

# Create subplots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# left: free energy heatmap
fe = meta.getFreeEnergy()
im = ax0.imshow(fe, extent=(-180, 180, -180, 180), origin='lower', cmap='viridis', aspect='auto')
ax0.set_xlabel('Phi (degrees)')
ax0.set_ylabel('Psi (degrees)')
ax0.set_title('Free Energy Landscape of Alanine Dipeptide in vacuo')
cbar = fig.colorbar(im, ax=ax0)
cbar.set_label('Free Energy (kJ/mol)')

# right: CV time series
ax1.scatter(df_cv['time(ps)'], np.rad2deg(df_cv['cv0']), s=8, alpha=0.7, label='phi')
ax1.scatter(df_cv['time(ps)'], np.rad2deg(df_cv['cv1']), s=8, alpha=0.7, label='psi')
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('CV values (degrees)')
ax1.set_title('Wt-metadynamics Collective Variables over Time')
ax1.legend()

plt.show()