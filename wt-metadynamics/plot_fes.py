import numpy as np
from analysis import read_hills_file, reconstruct_fes, project_fes_1d
from plotting import plot_fes_2d, plot_fes_1d, plot_deltaF_over_time
import matplotlib.pyplot as plt

# Main execution
if __name__ == '__main__':
    # Read HILLS file
    save_fes = False
    stride = 500
    clip_fes_to = 50 # kJ/mol, for better visualization

    filename = 'HILLS'

    time_ps, cv0_arr, cv1_arr, sigma_cv0, sigma_cv1, heights, bounds = read_hills_file(filename)

    print(f"Read {len(cv0_arr)} hills from {filename}")
    print(f"CV0 range: [{bounds['min_cv0']:.2f}, {bounds['max_cv0']:.2f}]")
    print(f"CV1 range: [{bounds['min_cv1']:.2f}, {bounds['max_cv1']:.2f}]")

    # TODO: periodic vars here are hard-coded, need to read from header
    cv0_grid, cv1_grid, fes_arr = reconstruct_fes(
        cv0_arr, cv1_arr,
        sigma_cv0, sigma_cv1, heights,
        bounds,
        grid_points=200,
        periodic_cv0=True,
        periodic_cv1=True,
        stride=stride,
    )

    # make one figure with 4 subplots:
    # (1) CVs over time, (2) FES heatmap, (3) 1D FES (projected along phi) and (4) deltaF over time

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flatten()

    # (1) CV time evolution
    ax0.scatter(time_ps, cv0_arr, s=8, alpha=0.7, label='phi')
    ax0.scatter(time_ps, cv1_arr, s=8, alpha=0.7, label='psi')
    ax0.set_xlabel('Time (ps)')
    ax0.set_ylabel('CV values (rad)')
    ax0.set_title('Phi and Psi Evolution Over Time')
    ax0.legend()

    # (2) Plot 2D FES (limit to 50 kJ/mol for better visualization)
    im = ax1.imshow(fes_arr[-1], extent=(-3.14, 3.14, -3.14, 3.14),
                    origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=clip_fes_to,
                    )
    ax1.set_xlabel('Phi (rad)')
    ax1.set_ylabel('Psi (rad)')
    ax1.set_title('Free Energy Landscape of Alanine Dipeptide in vacuo')
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Free Energy (kJ/mol)')

    # Reconstruct 1D FES by projecting along phi (integrate out psi)
    phi_grid_lst, fes_phi_lst = [], []
    delta_F_lst = []
    for fes in fes_arr:
        phi_grid, fes_phi = project_fes_1d(cv0_grid, cv1_grid, fes,
                                           kT=2.494,  # 300K in kJ/mol
                                           project_along='cv0')
        phi_grid_lst.append(phi_grid)
        fes_phi_lst.append(fes_phi)
        #  the alpha and beta basins
        alpha_basin_fe, beta_basin_fe = np.min(fes_phi[50:75]), np.min(fes_phi[100:175])
        deltaF = alpha_basin_fe - beta_basin_fe
        delta_F_lst.append(deltaF)

    # (3) plot 1D FES, with one curve for each snapshot/stride interval
    ax2.set_title('1D Free Energy Profile (Projected Along Phi)', fontsize=12)
    for i, fes_1d in enumerate(fes_phi_lst):
        ax2.plot(phi_grid_lst[i], fes_1d, label=f'Hills {stride*i}-{stride*(i+1)}', linewidth=2)
    ax2.legend()
    ax2.set_xlabel('CV (rad)', fontsize=10)
    ax2.set_ylabel('Free Energy (kJ/mol)', fontsize=10)
    ax2.set_title('1D Free Energy Profile', fontsize=12)

    # plot deltaF over time
    ax3.plot(np.arange(len(delta_F_lst)) * stride, delta_F_lst, 'r-', linewidth=2,
             label=f'final ΔF = {delta_F_lst[-1]:.2f} kJ/mol)')
    ax3.legend()
    ax3.set_xlabel('Time (ps)', fontsize=10)
    ax3.set_ylabel('ΔF (kJ/mol)', fontsize=10)
    ax3.set_title('Free Energy Difference Between Alpha and Beta Basins', fontsize=12)

    plt.show()
