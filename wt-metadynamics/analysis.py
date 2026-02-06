import numpy as np


def read_hills_file(filename):
    """Read PLUMED HILLS file and extract header info and hills data."""

    # Read header lines to get CV bounds
    min_cv0, max_cv0 = None, None
    min_cv1, max_cv1 = None, None

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#! SET min_cv0'):
                min_cv0 = float(line.split()[-1])
            elif line.startswith('#! SET max_cv0'):
                max_cv0 = float(line.split()[-1])
            elif line.startswith('#! SET min_cv1'):
                min_cv1 = float(line.split()[-1])
            elif line.startswith('#! SET max_cv1'):
                max_cv1 = float(line.split()[-1])

    # Read hills data (skip comment lines)
    data = np.loadtxt(filename, comments=['#', '!'])

    # Extract columns: time, cv0, cv1, sigma_cv0, sigma_cv1, height, biasf
    cv0_centers = data[:, 1]
    cv1_centers = data[:, 2]
    sigma_cv0 = data[:, 3]
    sigma_cv1 = data[:, 4]
    heights = data[:, 5]

    bounds = {
        'min_cv0': min_cv0,
        'max_cv0': max_cv0,
        'min_cv1': min_cv1,
        'max_cv1': max_cv1
    }

    return cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds


def periodic_distance(x, x0, period):
    """Calculate periodic distance between x and x0."""
    delta = x - x0
    # Shift to [-period/2, period/2]
    delta = delta - period * np.round(delta / period)
    return delta


def gaussian_2d(
        x, y, x0, y0, sigma_x, sigma_y, height, period_x=None, period_y=None,
):
    """2D Gaussian function with periodic boundary conditions."""

    # Calculate distances
    if period_x is not None:
        dx = periodic_distance(x, x0, period_x)
    else:
        dx = x - x0

    if period_y is not None:
        dy = periodic_distance(y, y0, period_y)
    else:
        dy = y - y0

    return height * np.exp(-0.5 * (dx ** 2 / sigma_x ** 2 + dy ** 2 / sigma_y ** 2))


# TODO: number of CVs in FES here is hard-coded to 2, need to generalise
def reconstruct_fes(
        cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights,
        bounds, grid_points=200, periodic_cv0=False, periodic_cv1=False,
        stride=1,
):
    """Reconstruct FES progressively, computing snapshots every
    'stride' hills.

    Parameters:
    -----------
    stride : int
        Compute FES every 'stride' hills (e.g., stride=10 gives FES
        at hills 10, 20, 30, ...)

    Returns:
    --------
    cv0_grid : array
        CV0 grid points
    cv1_grid : array
        CV1 grid points
    fes_snapshots : array
        array of the FES, shape (n_snapshots, grid_points, grid_points),
        where n_snapshots is the number of FES snapshots computed every
        'stride' hills.
    """

    # Create grid
    cv0_grid = np.linspace(bounds['min_cv0'], bounds['max_cv0'], grid_points)
    cv1_grid = np.linspace(bounds['min_cv1'], bounds['max_cv1'], grid_points)
    CV0, CV1 = np.meshgrid(cv0_grid, cv1_grid)

    # Determine periods
    period_cv0 = (bounds['max_cv0'] - bounds['min_cv0']) if periodic_cv0 else None
    period_cv1 = (bounds['max_cv1'] - bounds['min_cv1']) if periodic_cv1 else None

    # Initialize bias potential
    bias = np.zeros_like(CV0)
    fes_snapshots = []

    n_hills = len(cv0_centers)
    print(f"Computing FES snapshots every {stride} hills (total {n_hills} hills)...")

    for i in range(n_hills):
        bias += gaussian_2d(
            CV0, CV1, cv0_centers[i], cv1_centers[i],
            sigma_cv0[i], sigma_cv1[i], heights[i],
            period_cv0, period_cv1
        )

        # Save snapshot every 'stride' hills
        if (i + 1) % stride == 0:
            fes = -bias.copy()
            fes -= np.min(fes)
            fes_snapshots.append(fes)

    return cv0_grid, cv1_grid, np.array(fes_snapshots)


def project_fes_1d(cv0_grid, cv1_grid, fes, kT=2.494, project_along='cv0'):
    """
    Project 2D FES onto 1D by integrating out one dimension.

    Parameters:
    -----------
    cv0_grid : array
        Grid points for CV0
    cv1_grid : array
        Grid points for CV1
    fes : 2D array
        Free energy surface
    kT : float
        Thermal energy in kJ/mol (default: 2.494 for 300K)
    project_along : str
        'cv0' to project along CV0 (integrate out CV1)
        'cv1' to project along CV1 (integrate out CV0)

    Returns:
    --------
    grid_1d : array
        1D grid
    fes_1d : array
        1D free energy profile
    """

    # Convert FES to probability (Boltzmann factor)
    prob = np.exp(-fes / kT)

    if project_along == 'cv0':
        # Integrate over CV1 (psi) to get F(phi)
        prob_1d = np.trapezoid(prob, cv1_grid, axis=0)
        grid_1d = cv0_grid
    elif project_along == 'cv1':
        # Integrate over CV0 (phi) to get F(psi)
        prob_1d = np.trapezoid(prob, cv0_grid, axis=1)
        grid_1d = cv1_grid
    else:
        raise ValueError("project_along must be 'cv0' or 'cv1'")

    # Convert back to free energy
    fes_1d = -kT * np.log(prob_1d)

    # Shift minimum to zero
    fes_1d -= np.min(fes_1d)

    return grid_1d, fes_1d
