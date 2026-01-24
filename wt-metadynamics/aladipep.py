from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import kelvin, kilojoules_per_mole, picosecond
from metadynamics import Metadynamics, BiasVariable
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
    remainingTime=True,speed=True,totalSteps=500000, separator=' ')
)

# TODO: write a reporter class to write the biased CVs to a file + the hill heights
# Create PLUMED compatible HILLS file.
file = open('HILLS','w')
file.write('#! FIELDS time pp.proj pp.ext sigma_pp.proj sigma_pp.ext height biasf\n')
file.write('#! SET multivariate false\n')
file.write('#! SET kerneltype gaussian\n')

# Initialise the collective variable array.
current_cvs = np.array(list(meta.getCollectiveVariables(simulation)) + [meta.getHillHeight(simulation)])

# Write the inital collective variable record.
colvar_array = np.array([current_cvs])
line = colvar_array[0]
time = 0
write_line = f'{time:15} {line[0]:20.16f} {line[1]:20.16f}          {sigma_cv0}           {sigma_cv1} {line[2]:20.16f}            {bias_f}\n'
file.write(write_line)

# Run the simulation.
steps_per_ns = 500000
sim_time_ns = 1
total_steps = int(sim_time_ns * steps_per_ns)

steps_per_cycle = 1000
total_cycles = math.ceil(total_steps / steps_per_cycle)
remaining_steps = int(sim_time_ns * steps_per_ns)
remaining_cycles = math.ceil(remaining_steps / steps_per_cycle)
start_cycles = total_cycles - remaining_cycles
for cycle in range(start_cycles, total_cycles):
    meta.step(simulation, steps_per_cycle)
    current_cvs = np.array(list(meta.getCollectiveVariables(simulation)) + [meta.getHillHeight(simulation)])
    colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save('COLVAR.npy', colvar_array)
    line = colvar_array[cycle + 1]
    time = int((cycle + 1) * 0.002*steps_per_cycle)
    write_line = f'{time:15} {line[0]:20.16f} {line[1]:20.16f}          {sigma_cv0}           {sigma_cv1} {line[2]:20.16f}            {bias_f}\n'
    file.write(write_line)
file.close()

# TODO: flip the axes to match conventional representation of phi/psi
# Create a contour plot of the free energy landscape.
plt.imshow(meta.getFreeEnergy())
plt.show()
