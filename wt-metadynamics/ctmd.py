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


def main(args):
    """
    args.structure : str, default='solvated.rst7'
        Name of the structure file, either Amber or Gromacs format.
    args.parameters : str, default='solvated.prm7'
        Name of the parameter or topology file, either Amber or Gromacs
        format.
    args.output : str, default='.'
        Path to and the name of the output directory.
    args.lig_resname : str, default='LIG'
        Residue name of the ligand in the structure/parameter file.
    args.nreps : int, default=10
        Number of repeat OpenBPMD simulations to run in series.
    args.hill_height : float, default=0.3
        Size of the metadynamical hill, in kcal/mol.
    """
    if args.structure.endswith('.gro'):
        coords = GromacsGroFile(args.structure)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(args.parameters, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(args.structure)
        parm = AmberPrmtopFile(args.parameters)

    if not os.path.isdir(f'{args.output}'):
        os.mkdir(f'{args.output}')

    # Run NREPS number of production simulations
    for idx in range(0, args.nreps):
        rep_dir = os.path.join(args.output, f'rep_{idx}')
        if not os.path.isdir(rep_dir):
            os.mkdir(rep_dir)

        if os.path.isfile(os.path.join(rep_dir, 'ctmd_results.csv')):
            continue

        produce(args.output, idx, args.lig_resname, coords, parm, args.parameters,
                args.structure, 0.416666667, anchor_atom_lst=[])


    return None


def compute_ct(s_grid, V_bias, beta, gamma):
    """
    Compute c(t) from equation 1 (Tiwary & Parrinello 2015)

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

    # c(t) = (1/beta) * log(numerator / denominator)
    c_t = (1 / beta) * np.log(numerator / denominator)

    return c_t


def produce(out_dir, idx, lig_resname, input_pos, parm, parm_file,
            coords_file, set_hill_height, anchor_atom_lst: list = []):
    """An CTMD production simulation function. Ligand RMSD is biased with
    wt-metadynamics. The integrator uses a 4 fs time step and
    runs for 5 ns, writing a frame every 100 ps.

    Writes a 'trj.dcd', 'COLVAR.npy', 'bias_*.npy' and 'sim_log.csv' files
    during the metadynamics simulation in the '{out_dir}/rep_{idx}' directory.
    After the simulation is done, it analyses the trajectories and writes a
    'ctmd_results.csv' file.

    Parameters
    ----------
    out_dir : str
        Directory where your equilibration PDBs and 'rep_*' dirs are at.
    idx : int
        Current replica index.
    lig_resname : str
        Residue name of the ligand.
    input_pos : AmberInpcrdFile or GromacsGroFile object
        Name of the PDB for equilibrated system.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    parm_file : str
        The name of the parameter or topology file of the system.
    set_hill_height : float
        Metadynamic hill height, in kcal/mol.
    anchor_atom_lst : list, default=[]
        List of atom indices to use as anchor atoms for the RMSD calculation.
         If empty, the function will automatically select heavy backbone atoms
         within 10 angstroms of the protein's center of mass as anchor atoms.
    """
    # First, assign the replica directory to which we'll write the files
    write_dir = os.path.join(out_dir, f'rep_{idx}')

    lig_ha_lst = [
        atom.index for atom in topology.atoms()
        if atom.residue.name == lig_resname and not atom.name.startswith("H")
    ]

    # Set up the system to run metadynamics
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometer,
        constraints=HBonds,
        hydrogenMass=4 * amu
    )
    # get the atom positions for the system from the equilibrated
    # system
    input_positions = input_pos.getPositions()

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

    rmsd = RMSDForce(input_positions, alignment_indices)
    # Set up the typical metadynamics parameters
    grid_min, grid_max = 0.0, 1.0  # nm
    hill_height = set_hill_height * kilocalories_per_mole
    hill_width = 0.015  # nm, also known as sigma

    grid_width = hill_width / 5
    # 'grid' here refers to the number of grid points
    grid = int(abs(grid_min - grid_max) / grid_width)

    rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width,
                           False, gridWidth=grid)

    # define the metadynamics object
    # deposit bias every 1 ps, BF = 4, write bias every ns
    meta = Metadynamics(system, [rmsd_cv], 300.0 * kelvin, 10.0, hill_height,
                        250, biasDir=write_dir,
                        saveFrequency=250000)

    # Set up and run metadynamics
    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond,
                                    0.004 * picosecond)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}

    simulation = Simulation(parm.topology, system, integrator, platform,
                            properties)
    simulation.context.setPositions(input_positions)

    trj_name = os.path.join(write_dir, 'trj.dcd')

    sim_time = 5  # ns
    steps = 250000 * sim_time

    simulation.reporters.append(DCDReporter(trj_name, 25000))  # every 100 ps
    simulation.reporters.append(StateDataReporter(
        os.path.join(write_dir, 'sim_log.csv'), 250000,
        step=True, temperature=True, progress=True,
        remainingTime=True, speed=True,
        totalSteps=steps, separator=','))  # every 1 ns

    colvar_array = np.array([meta.getCollectiveVariables(simulation)])
    # calculate the c(t) as described in Eq 1. of https://www.biorxiv.org/content/10.64898/2026.02.05.703972v1.full.pdf
    c_t = compute_ct(meta._grid, meta._bias, 1 / (0.008314 * 300), meta._biasFactor)
    print(f"Initial c(t): {c_t:.2f} kJ/mol")
    for i in range(0, int(steps), 500):
        c_t = compute_ct(meta._grid, meta._bias, 1 / (0.008314 * 300), meta._biasFactor)
        print(f"step {i}: c(t): {c_t:.2f} kJ/mol")
        if i % 25000 == 0:
            # log the stored COLVAR every 100ps
            np.save(os.path.join(write_dir, 'COLVAR.npy'), colvar_array)
        meta.step(simulation, 500)
        current_cvs = meta.getCollectiveVariables(simulation)
        # record the CVs every 2 ps
        colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save(os.path.join(write_dir, 'COLVAR.npy'), colvar_array)

    return None
