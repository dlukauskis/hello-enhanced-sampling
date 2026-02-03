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

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, height):
    """2D Gaussian function."""
    return height * np.exp(-0.5 * ((x - x0)**2 / sigma_x**2 + (y - y0)**2 / sigma_y**2))

# TODO: produces the right FES shape, but does not understand periodicity, fix that
# TODO: also, the scale is off by a factor of 10, investigate
def reconstruct_fes(cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, 
                    bounds, grid_points=200):
    """Reconstruct the free energy surface by summing all Gaussians."""
    
    # Create grid
    cv0_grid = np.linspace(bounds['min_cv0'], bounds['max_cv0'], grid_points)
    cv1_grid = np.linspace(bounds['min_cv1'], bounds['max_cv1'], grid_points)
    CV0, CV1 = np.meshgrid(cv0_grid, cv1_grid)
    
    # Initialize bias potential
    bias = np.zeros_like(CV0)
    
    # Sum all Gaussian hills
    n_hills = len(cv0_centers)
    for i in range(n_hills):
        bias += gaussian_2d(CV0, CV1, 
                           cv0_centers[i], cv1_centers[i],
                           sigma_cv0[i], sigma_cv1[i],
                           heights[i])
    
    # Free energy = -bias
    fes = -bias
    
    # Shift minimum to zero
    fes -= np.min(fes)
    
    return cv0_grid, cv1_grid, fes

def plot_fes(cv0_grid, cv1_grid, fes, max_energy=None):
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
    plt.savefig('fes.png', dpi=300)
    plt.show()

# Main execution
if __name__ == '__main__':
    # Read HILLS file
    filename = 'HILLS'
    cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds = read_hills_file(filename)
    
    print(f"Read {len(cv0_centers)} hills from {filename}")
    print(f"CV0 range: [{bounds['min_cv0']:.2f}, {bounds['max_cv0']:.2f}]")
    print(f"CV1 range: [{bounds['min_cv1']:.2f}, {bounds['max_cv1']:.2f}]")
    
    # Reconstruct FES
    cv0_grid, cv1_grid, fes = reconstruct_fes(cv0_centers, cv1_centers, 
                                               sigma_cv0, sigma_cv1, heights, 
                                               bounds, grid_points=200)
    
    # Save FES to file
    np.savetxt('fes.dat', np.column_stack([cv0_grid[:, None].repeat(len(cv1_grid), axis=1).flatten(),
                                           cv1_grid[None, :].repeat(len(cv0_grid), axis=0).flatten(),
                                           fes.flatten()]),
               header='cv0 cv1 fes(kJ/mol)')
    
    # Plot (limit to 50 kJ/mol for better visualization)
    plot_fes(cv0_grid, cv1_grid, fes, max_energy=50)
    
    print(f"FES saved to fes.dat and fes.png")
