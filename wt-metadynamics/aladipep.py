from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond
from metadynamics import Metadynamics, BiasVariable
from metadynamicsreporter import MetadynamicsReporter
import numpy as np
import matplotlib.pyplot as plt

# Create a System for alanine dipeptide in water.
total_steps = 500000

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

# Create a contour plot of the free energy landscape.
plt.imshow(
    meta.getFreeEnergy(),
    extent=(-180, 180, -180, 180),
    origin='lower', cmap='viridis',
)
plt.colorbar(label='Free Energy (kJ/mol)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.title('Free Energy Landscape of Alanine Dipeptide in vacuo')
plt.show()
