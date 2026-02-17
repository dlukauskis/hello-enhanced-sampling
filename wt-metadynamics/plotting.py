import matplotlib.pyplot
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

    plt.xlabel('CV0 (units)')
    plt.ylabel('CV1 (units)')
    plt.title('Free Energy Surface')
    plt.tight_layout()
    if save_plot:
        plt.savefig('fes.png', dpi=300)
    plt.show()


def plot_fes_1d(
        grid: list | np.ndarray,
        fes_1d: list | np.ndarray,
        label='CV (rad)',
        max_energy=50,
        save_plot=False,
        filename='fes_1d.png'
):
    """Plot 1D free energy profile."""

    if max_energy is not None:
        if type(fes_1d) is list:
            fes_plot = []
            for fes in fes_1d:
                cliped_fes_1d = np.clip(fes, 0, max_energy)
                fes_plot.append(cliped_fes_1d)
        else:
            fes_plot = np.clip(fes_1d, 0, max_energy)
    else:
        fes_plot = fes_1d

    plt.figure(figsize=(10, 6))
    if type(fes_plot) is list:
        for i, fes in enumerate(fes_plot):
            plt.plot(grid[i], fes, label=f'Snapshot {i+1}', linewidth=2)
        plt.legend()
    else:
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

def plot_deltaF_over_time(delta_F_lst, stride=500, save_plot=False, filename='deltaF_over_time.png'):
    """Plot the free energy difference between alpha and beta basins over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(delta_F_lst)) * stride, delta_F_lst, 'r-', linewidth=2)
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('ΔF (kJ/mol)', fontsize=12)
    plt.title('Free Energy Difference Between Alpha and Beta Basins Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=300)
    plt.show()
