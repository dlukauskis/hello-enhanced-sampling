from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond
from metadynamics import Metadynamics, BiasVariable
from metadynamicsreporter import MetadynamicsReporter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import project_fes_1d, reconstruct_fes, read_hills_file
from plotting import plot_fes_1d

# Simulation parameters
total_steps = 500000
hills_pace = 500
hills_write_pace = 500
bias_f = 6


# Create a System for alanine dipeptide in vacuo
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
meta = Metadynamics(
    system, [phi, psi], 300.0 * kelvin,
    bias_f, 1.2 * kilojoules_per_mole, hills_pace,
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
    MetadynamicsReporter('HILLS', hills_write_pace)
)

meta.step(simulation, total_steps)

# Plot CV time series
df_cv = pd.read_csv('HILLS', delimiter=r'\s+', skiprows=7, header=None)
df_cv.columns = ['time(ps)', 'cv0', 'cv1', 'sigma_cv0', 'sigma_cv1', 'hill_height', 'bias_factor']

# Create subplots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# left: free energy heatmap
fes = meta.getFreeEnergy()
fes -= np.min(fes)  # shift to zero
im = ax0.imshow(fes, extent=(-180, 180, -180, 180), origin='lower', cmap='viridis', aspect='auto')
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

plt.savefig('fes_cvs.png', dpi=300)
plt.show()

# TODO: plot the free energy convergence over time by integrating the hills
stride = 500
clip_fes_to = 50  # kJ/mol, for better visualization
cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds = read_hills_file('HILLS')

cv0_grid, cv1_grid, fes_arr = reconstruct_fes(
    cv0_centers, cv1_centers,
    sigma_cv0, sigma_cv1, heights,
    bounds,
    grid_points=200,
    periodic_cv0=True,
    periodic_cv1=True,
    stride=stride,
)

# Reconstruct 1D FES
# Project along phi (integrate out psi)
stride = 500  # number of hills
phi_grid_lst, fes_phi_lst = [], []
for fes in fes_arr:
    phi_grid, fes_phi = project_fes_1d(cv0_grid, cv1_grid, fes,
                                       kT=2.494,  # 300K in kJ/mol
                                       project_along='cv0')
    phi_grid_lst.append(phi_grid)
    fes_phi_lst.append(fes_phi)
# get these to plot in the same figure, with a label for each curve
# TODO: finish this, want as a subplot next to the 2D FES and CV time series,
#  showing how the 1D FES converges over time
plot_fes_1d(phi_grid_lst, fes_phi_lst, max_energy=clip_fes_to, save_plot=False)
