from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond
from metadynamics import Metadynamics, BiasVariable
from metadynamicsreporter import MetadynamicsReporter
from ctmd import compute_ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation parameters
total_steps = 50000
hills_pace = 500
hills_write_pace = 500
bias_f = 6


# Helper function to marginalize 2D bias to 1D and compute c(t)
def compute_ct_1d(meta, cv_index):
    """Marginalize 2D bias to 1D along specified CV and compute c(t)

    Proper marginalization: P(s_i) = integral of exp(-beta*V(s_i, s_j)) ds_j
    This requires taking the minimum free energy along the other dimension
    """
    beta = 1 / (0.008314 * 300)
    bias_2d = meta._totalBias

    # Marginalize: take minimum along the other CV axis
    # (Free energy = -1/beta * log(integral of exp(-beta*V)))
    if cv_index == 0:
        # Marginalize over CV1 (psi), keeping CV0 (phi)
        # Find minimum free energy along psi for each phi value
        bias_1d = np.min(bias_2d, axis=1)
    else:
        # Marginalize over CV0 (phi), keeping CV1 (psi)
        # Find minimum free energy along phi for each psi value
        bias_1d = np.min(bias_2d, axis=0)

    # Create grid for the marginalized CV
    cv = meta.variables[cv_index]
    s_grid = np.linspace(cv.minValue, cv.maxValue, len(bias_1d))

    return compute_ct(s_grid, bias_1d, beta, meta.biasFactor)



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

# Compute initial c(t) for both CVs (before simulation)
print(f"Initial c(t) for phi: {compute_ct_1d(meta, 0):.4f}")
print(f"Initial c(t) for psi: {compute_ct_1d(meta, 1):.4f}")

meta.step(simulation, total_steps)

# Compute final c(t) for both CVs (after simulation)
print(f"Final c(t) for phi: {compute_ct_1d(meta, 0):.4f}")
print(f"Final c(t) for psi: {compute_ct_1d(meta, 1):.4f}")

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

