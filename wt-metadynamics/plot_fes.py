import numpy as np
from analysis import read_hills_file, reconstruct_fes, project_fes_1d
from plotting import plot_fes_2d, plot_fes_1d

# Main execution
if __name__ == '__main__':
    # Read HILLS file
    save_fes = False
    fes_1d = True
    stride = 500
    clip_fes_to = 50 # kJ/mol, for better visualization

    filename = 'HILLS'

    cv0_centers, cv1_centers, sigma_cv0, sigma_cv1, heights, bounds = read_hills_file(filename)
    
    print(f"Read {len(cv0_centers)} hills from {filename}")
    print(f"CV0 range: [{bounds['min_cv0']:.2f}, {bounds['max_cv0']:.2f}]")
    print(f"CV1 range: [{bounds['min_cv1']:.2f}, {bounds['max_cv1']:.2f}]")

    # TODO: periodic vars here are hard-coded, need to read from header
    cv0_grid, cv1_grid, fes_arr = reconstruct_fes(
        cv0_centers, cv1_centers,
        sigma_cv0, sigma_cv1, heights,
        bounds,
        grid_points=200,
        periodic_cv0=True,
        periodic_cv1=True,
        stride=stride,
    )
    if not stride:
        fes = fes_arr[-1]
    # Reconstruct 1D FES
    if fes_1d:
        # Project along phi (integrate out psi)
        if stride:
            phi_grid_lst, fes_phi_lst = [], []
            for fes in fes_arr:
                phi_grid, fes_phi = project_fes_1d(cv0_grid, cv1_grid, fes,
                                                   kT=2.494,  # 300K in kJ/mol
                                                   project_along='cv0')
                phi_grid_lst.append(phi_grid)
                fes_phi_lst.append(fes_phi)
            print(f"Projected 1D FES shape: {phi_grid.shape} {fes_phi.shape}")
            print(f"Types: {type(phi_grid)} {type(fes_phi)}")
            # get these to plot in the same figure, with a label for each curve
            plot_fes_1d(phi_grid_lst, fes_phi_lst, max_energy=clip_fes_to, save_plot=False)
        else:
            phi_grid, fes_phi = project_fes_1d(cv0_grid, cv1_grid, fes,
                                               kT=2.494,  # 300K in kJ/mol
                                               project_along='cv0')
            plot_fes_1d(phi_grid, fes_phi, max_energy=clip_fes_to, save_plot=False)

    else:
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
        plot_fes_2d(cv0_grid, cv1_grid, fes, max_energy=clip_fes_to, save_plot=save_fes)
