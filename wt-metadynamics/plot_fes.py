import numpy as np
import matplotlib.pyplot as plt


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
    
    # Extract columns: step, time, cv0, cv1, sigma_cv0, sigma_cv1, height, biasf
    cv0_centers = data[:, 2]
    cv1_centers = data[:, 3]
    sigma_cv0 = data[:, 4]
    sigma_cv1 = data[:, 5]
    heights = data[:, 6]
    
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
    ):
    """Reconstruct the free energy surface by summing all Gaussians.

    Parameters:
    -----------
    periodic_cv0 : bool
        Whether CV0 is periodic (e.g., dihedral angle)
    periodic_cv1 : bool
        Whether CV1 is periodic (e.g., dihedral angle)
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

    # Sum all Gaussian hills
    n_hills = len(cv0_centers)
    print(f"Summing {n_hills} Gaussian hills...")

    for i in range(n_hills):
        if (i + 1) % 1000 == 0:
            print(f"  Processing hill {i + 1}/{n_hills}")

        bias += gaussian_2d(
            CV0, CV1, cv0_centers[i], cv1_centers[i],
            sigma_cv0[i], sigma_cv1[i], heights[i],
            period_cv0, period_cv1
        )

    fes = -bias

    # Shift minimum to zero
    fes -= np.min(fes)

    return cv0_grid, cv1_grid, fes


def plot_fes(cv0_grid, cv1_grid, fes, max_energy=None, save_plot=False):
    """Plot the free energy surface."""
    
    if max_energy is not None:
        fes_plot = np.clip(fes, 0, max_energy)
    else:
        fes_plot = fes
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(cv0_grid, cv1_grid, fes_plot, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Free Energy (kJ/mol)')
    plt.contour(cv0_grid, cv1_grid, fes_plot, levels=10, colors='white', 
                linewidths=0.5, alpha=0.5)
    
    plt.xlabel('CV0 (rad)')
    plt.ylabel('CV1 (rad)')
    plt.title('Free Energy Surface')
    plt.tight_layout()
    if save_plot:
        plt.savefig('fes.png', dpi=300)
    plt.show()


def compute_1d_fes_by_marginalization(
    cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds,
    grid_points=200, grid_points_other=None,
    periodic_cv0=False, periodic_cv1=False,
    analytic_when_possible=True
):
    """
    Return (cv0_grid, fes1d) by marginalizing over CV1 (numeric integration).
    If analytic_when_possible=True and CV1 is non-periodic, use analytic integral
    of a Gaussian over the full real line for speed.
    """
    if grid_points_other is None:
        grid_points_other = grid_points

    # grids
    cv0_grid = np.linspace(bounds['min_cv0'], bounds['max_cv0'], grid_points)
    cv1_grid = np.linspace(bounds['min_cv1'], bounds['max_cv1'], grid_points_other)
    dcv1 = cv1_grid[1] - cv1_grid[0]

    period_cv0 = (bounds['max_cv0'] - bounds['min_cv0']) if periodic_cv0 else None
    period_cv1 = (bounds['max_cv1'] - bounds['min_cv1']) if periodic_cv1 else None

    bias1d = np.zeros_like(cv0_grid)

    n_hills = len(cv0_centers)
    for i in range(n_hills):
        x0 = cv0_centers[i]
        y0 = cv1_centers[i]
        sx = sigma_cv0[i]
        sy = sigma_cv1[i]
        h = heights[i]

        # Use analytic marginalization over y if allowed and non-periodic:
        if analytic_when_possible and (period_cv1 is None):
            # Integral over y (−inf..inf) of h * exp(-0.5*(dx^2/sx^2 + dy^2/sy^2)) dy
            # = h * sqrt(2*pi) * sy * exp(-0.5 * dx^2 / sx^2)
            dx = cv0_grid - x0
            if period_cv0 is not None:
                # apply periodic distance along cv0
                dx = dx - period_cv0 * np.round(dx / period_cv0)
            contribution = h * np.sqrt(2.0 * np.pi) * sy * np.exp(-0.5 * (dx ** 2 / sx ** 2))
            bias1d += contribution
        else:
            # Numeric marginalization: evaluate gaussian on (cv0 x cv1) mesh and sum over cv1
            # vectorized per hill
            CV0 = cv0_grid[:, None]          # shape (n_cv0, 1)
            CV1 = cv1_grid[None, :]          # shape (1, n_cv1)
            # periodic-aware distance computation (reuse periodic_distance logic)
            dx = CV0 - x0
            if period_cv0 is not None:
                dx = dx - period_cv0 * np.round(dx / period_cv0)
            dy = CV1 - y0
            if period_cv1 is not None:
                dy = dy - period_cv1 * np.round(dy / period_cv1)
            gauss = h * np.exp(-0.5 * (dx ** 2 / sx ** 2 + dy ** 2 / sy ** 2))
            # sum over cv1 axis and multiply by dcv1 to approximate integral
            contribution = np.sum(gauss, axis=1) * dcv1
            bias1d += contribution

    fes1d = -bias1d
    fes1d -= np.min(fes1d)
    return cv0_grid, fes1d


def plot_1d_fes(cv0_grid, fes1d, max_energy=None, save_plot=False, filename='fes_1d.png'):
    """Simple 1D plot of FES vs CV0."""
    if max_energy is not None:
        fes_plot = np.clip(fes1d, 0, max_energy)
    else:
        fes_plot = fes1d

    plt.figure(figsize=(8, 4))
    plt.plot(cv0_grid, fes_plot, '-k')
    plt.fill_between(cv0_grid, fes_plot, alpha=0.2)
    plt.xlabel('CV0 (phi)')
    plt.ylabel('Free Energy (kJ/mol)')
    plt.title('1D FES (marginalized over CV1)')
    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=300)
    plt.show()


# Main execution
if __name__ == '__main__':
    # Read HILLS file
    save_fes = False
    fes_1d =True
    filename = 'HILLS'

    cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds = read_hills_file(filename)
    
    print(f"Read {len(cv0_centers)} hills from {filename}")
    print(f"CV0 range: [{bounds['min_cv0']:.2f}, {bounds['max_cv0']:.2f}]")
    print(f"CV1 range: [{bounds['min_cv1']:.2f}, {bounds['max_cv1']:.2f}]")

    # TODO: periodic vars here are hard-coded, need to read from header
    # Reconstruct 1D FES
    if fes_1d:
        # TODO: somehow this didn't quite work, there is one minimum at around -2.5 to -
        #  1.2 radians of phi (expected, correct), but missing the other minimum at around +1 to +1.5
        cv0_grid_1d, fes1d = compute_1d_fes_by_marginalization(
            cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds,
            grid_points=300, periodic_cv0=True, periodic_cv1=True,
        )
        plot_1d_fes(cv0_grid_1d, fes1d, max_energy=50)


    else:
        cv0_grid, cv1_grid, fes = reconstruct_fes(
            cv0_centers, cv1_centers,
            sigma_cv0, sigma_cv1, heights,
            bounds,
            grid_points=200,
            periodic_cv0=True,
            periodic_cv1=True,
        )

        # Save FES to file
        if save_fes:
            np.savetxt(
                'fes.dat',
                np.column_stack([cv0_grid[:, None].repeat(len(cv1_grid), axis=1).flatten(),
                                 cv1_grid[None, :].repeat(len(cv0_grid), axis=0).flatten(),
                                 fes.flatten()]),
               header='cv0 cv1 fes(kJ/mol)',
            )

        # Plot (limit to 50 kJ/mol for better visualization)
        plot_fes(cv0_grid, cv1_grid, fes, max_energy=50, save_plot=save_fes)
