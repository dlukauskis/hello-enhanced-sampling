import matplotlib.pyplot as plt
import numpy as np


def plot_fes_2d(cv0_grid, cv1_grid, fes, max_energy=None, save_plot=False):
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


def plot_fes_1d(grid, fes_1d: list | np.ndarray, label='CV (rad)', max_energy=50, save_plot=False, filename='fes_1d.png'):
    """Plot 1D free energy profile."""
    # if type(fes_1d) is list:

    if max_energy is not None:
        fes_plot = np.clip(fes_1d, 0, max_energy)
    else:
        fes_plot = fes_1d

    plt.figure(figsize=(10, 6))
    plt.plot(grid, fes_plot, 'b-', linewidth=2)
    plt.xlabel(label, fontsize=12)
    plt.ylabel('Free Energy (kJ/mol)', fontsize=12)
    plt.title('1D Free Energy Profile', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Set ticks for dihedral angles
    if 'rad' in label.lower():
        plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                   ['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=300)
    plt.show()
