from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond
from openmm.app.metadynamics import Metadynamics, BiasVariable
import numpy as np
import matplotlib.pyplot as plt

# Create a System for alanine dipeptide in water.

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
phi = BiasVariable(cv1, -np.pi, np.pi, 0.35, True)
cv2 = CustomTorsionForce('theta')
cv2.addTorsion(6, 8, 14, 16)
psi = BiasVariable(cv2, -np.pi, np.pi, 0.35, True)

# Set up the simulation.
meta = Metadynamics(
    system, [phi, psi], 300.0 * kelvin,
    6, 1.2 * kilojoules_per_mole, 100
)
integrator = LangevinIntegrator(
    300 * kelvin, 1.0 / picosecond, 0.002 * picosecond
)
simulation = Simulation(pdb_file.topology, system, integrator)
simulation.context.setPositions(pdb_file.positions)

simulation.reporters.append(StateDataReporter(
    stdout, 50000, step=True,
    temperature=True,progress=True,
    remainingTime=True,speed=True,totalSteps=500000, separator=' '))
# TODO: add a reporter to write the biased CVs to a file + the hill heights

# Run the simulation and plot the free energy landscape.
meta.step(simulation, 500000)

# TODO: flip the axes to match conventional representation of phi/psi
# Create a contour plot of the free energy landscape.
plt.imshow(meta.getFreeEnergy())
plt.show()
