#!/usr/bin/env python

# OpenMM
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.app.metadynamics import *

# The rest
import numpy as np
import os

__author__ = "Dominykas Lukauskis"
__version__ = "0.1.0"
__email__ = "lukauskisdominykas@gmail.com"

# TODO: move this function to metadynamics.py
def compute_ct(s_grid, V_bias, beta, gamma):
    """
    Compute c(t) from equation 1 (https://www.biorxiv.org/content/10.64898/2026.02.05.703972v1.full.pdf)

    Parameters:
    -----------
    s_grid : array
        Grid of collective variable values
    V_bias : array
        Metadynamics bias V(s,t) evaluated at each s_grid point
    beta : float
        Inverse temperature (1/kT) in 1/(kJ/mol)
    gamma : float
        Well-tempered bias factor

    Returns:
    --------
    c_t : float
        The c(t) value in kJ/mol
    """
    # Numerator: integral of exp[gamma/(gamma-1) * beta*V(s,t)]
    numerator = np.trapezoid(np.exp(gamma / (gamma - 1) * beta * V_bias), s_grid)

    # Denominator: integral of exp[1/(gamma-1) * beta*V(s,t)]
    denominator = np.trapezoid(np.exp(1 / (gamma - 1) * beta * V_bias), s_grid)

    c_t = (1 / beta) * np.log(numerator / denominator)

    return c_t

"""
structure : str
    Name of the structure file, either Amber or Gromacs format.
parameters : str
    Name of the parameter or topology file, either Amber or Gromacs
    format.
output : str
    Path to and the name of the output directory.
lig_resname
    Residue name of the ligand in the structure/parameter file.
hill_height : float, default=1.75
    Size of the metadynamical hill, in kJ/mol.
"""

structure = '../benchmark_systems/bCD-G1/system.gro'
parameters = '../benchmark_systems/bCD-G1/system.top'
out_dir_name, idx = 'tmp', 0
lig_resname = 'GST'
set_hill_height = 1.75
anchor_atom_lst = [0,21,42,63,84,105,126]  # for proteins, this can be the CA atoms near protein CoG
in_vacuo = True  # only for testing purposes with the host-guest system
sim_time = 5  # ns
deposition_pace = 250  # deposit a hill every 1 ps, so 250 steps with a 4 fs timestep


if structure.endswith('.gro'):
    coords = GromacsGroFile(structure)
    box_vectors = coords.getPeriodicBoxVectors()
    parm = GromacsTopFile(parameters, periodicBoxVectors=box_vectors)
else:
    coords = AmberInpcrdFile(structure)
    parm = AmberPrmtopFile(parameters)

# First, assign the replica directory to which we'll write the files
out_dir_path = os.path.join(out_dir_name, f'rep_{idx}')
os.makedirs(out_dir_path, exist_ok=True)

lig_ha_lst = [
    atom.index for atom in parm.topology.atoms()
    if atom.residue.name == lig_resname and not atom.name.startswith("H")
]

# Set up the system to run metadynamics
if in_vacuo:
    system = parm.createSystem(
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        hydrogenMass=4 * amu,
    )
else:
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        hydrogenMass=4 * amu,
    )

# get the atom positions for the system from the equilibrated
# system
input_positions = coords.getPositions()

# Add an 'empty' flat-bottom restraint to fix the issue with PBC.
# Without one, RMSDForce object fails to account for PBC.
k = 0 * kilojoules_per_mole  # NOTE - 0 kJ/mol constant
upper_wall = 10.00 * nanometer
fb_eq = '(k/2)*max(distance(g1,g2) - upper_wall, 0)^2'
upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
upper_wall_rest.addGroup(lig_ha_lst)
upper_wall_rest.addGroup(anchor_atom_lst)
upper_wall_rest.addBond([0, 1])
upper_wall_rest.addGlobalParameter('k', k)
upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
system.addForce(upper_wall_rest)

alignment_indices = lig_ha_lst + anchor_atom_lst

# OpenMM RMSD CV is not the same as the one in PLUMED, but it'll do
rmsd = RMSDForce(input_positions, alignment_indices)
# Set up the typical metadynamics parameters
grid_min, grid_max = 0.0, 1.0  # in nm
hill_height = set_hill_height * kilojoules_per_mole
hill_width = 0.015  # in nm, aka sigma

grid_width = hill_width / 5
# 'grid' here refers to the number of grid points
grid = int(abs(grid_min - grid_max) / grid_width)

rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width,
                       False, gridWidth=grid)

# define the metadynamics object, deposit bias every 1 ps, BF = 10, write bias every ns
meta = Metadynamics(
    system, [rmsd_cv], 300.0 * kelvin, 10.0, hill_height, deposition_pace,
    biasDir=out_dir_path, saveFrequency=250000
)

# Set up and run metadynamics
integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.004 * picosecond)

if in_vacuo:
    platform = Platform.getPlatformByName('CPU')
    properties = {}
else:
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}

simulation = Simulation(parm.topology, system, integrator, platform, properties)
simulation.context.setPositions(input_positions)

trj_fpath = os.path.join(out_dir_path, 'trj.dcd')
sim_log_fpath = os.path.join(out_dir_path, 'sim_log.csv')

total_steps = 250000 * sim_time

simulation.reporters.append(DCDReporter(trj_fpath, 25000))  # every 100 ps
simulation.reporters.append(StateDataReporter(
    sim_log_fpath, 250000,
    step=True, temperature=True, progress=True,
    remainingTime=True, speed=True,
    totalSteps=total_steps, separator=','))  # every 1 ns

colvar_fpath = os.path.join(out_dir_path, 'COLVAR.npy')

# Create grid for 1D bias
cv = meta.variables[0]
s_grid = np.linspace(cv.minValue, cv.maxValue, meta._totalBias.shape[0])
c_t = compute_ct(s_grid, meta._totalBias, 1 / (0.008314 * 300), meta.biasFactor)

starting_cv = meta.getCollectiveVariables(simulation)[0]
colvar_array = np.array([starting_cv, c_t])

ps_spent_above_6_ang = 0
for i in range(0, int(total_steps), deposition_pace):
    c_t = compute_ct(s_grid, meta._totalBias, 1 / (0.008314 * 300), meta.biasFactor)
    if i % 25000 == 0:
        # log the stored COLVAR every 100ps
        np.save(colvar_fpath, colvar_array)
    meta.step(simulation, deposition_pace)
    current_cv = meta.getCollectiveVariables(simulation)[0]
    if current_cv > 0.6:  # 0.6 nm = 6 angstroms
        ps_spent_above_6_ang += 1

    # record the CVs every 1 ps
    colvar_array = np.vstack([colvar_array, np.array([current_cv, c_t])])

    # if more than 200 ps are spent above 6 angstroms, stop the simulation according to protocol
    if ps_spent_above_6_ang > 200:
        break

# save the final COLVAR array to file
np.save(colvar_fpath, colvar_array)
