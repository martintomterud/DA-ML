import numpy as np
import matplotlib.pyplot as plt

import pde_solver


def part_b():

    # Set initial conditions
    dx_1 = 1/10
    dx_2 = 1/100

    Tmax_1 = 0.3
    Tmax_2 = 0.3

    # Sol 1, 1
    u_1, x_1, t_1 = pde_solver.forwardEuler(dx_1, Tmax_1)
    # Meshgrid and analytical solution
    X_1, T_1 = np.meshgrid(x_1, t_1)
    U_anlytic_1 = pde_solver.analyticalSolution(X_1, T_1)

    # Sol 2, 2
    u_2, x_2, t_2 = pde_solver.forwardEuler(dx_2, Tmax_2)
    # Meshgrid and analytical solution
    X_2, T_2 = np.meshgrid(x_2, t_2)
    U_anlytic_2 = pde_solver.analyticalSolution(X_2, T_2)

    # Compute relative errors
    rel_error_1 = np.abs(U_anlytic_1 - u_1) 
    meanRelEr_1 = np.mean(rel_error_1.flatten())

    rel_error_2 = np.abs(U_anlytic_2 - u_2) 
    meanRelEr_2 = np.mean(rel_error_2.flatten())

    # Set up figures

    fig1, axs1 = plt.subplots()
    pan1 = axs1.pcolormesh(X_1, T_1, u_1, cmap = 'jet')
    axs1.set_xlabel(r'$x$')
    axs1.set_ylabel(r'$t$')
    cbar = fig1.colorbar(pan1, ax = axs1)
    cbar.set_label(r'$u(x, t)$')

    fig1, axs1 = plt.subplots()
    pan1 = axs1.pcolormesh(X_1, T_1, rel_error_1, cmap = 'gist_ncar')
    axs1.set_xlabel(r'$x$')
    axs1.set_ylabel(r'$t$')
    cbar = fig1.colorbar(pan1, ax = axs1)
    cbar.set_label(r'Relative Error')

    fig1, axs1 = plt.subplots()
    pan1 = axs1.pcolormesh(X_2, T_2, u_2, cmap = 'jet')
    axs1.set_xlabel(r'$x$')
    axs1.set_ylabel(r'$t$')
    cbar = fig1.colorbar(pan1, ax = axs1)
    cbar.set_label(r'$u(x, t)$')

    fig1, axs1 = plt.subplots()
    pan1 = axs1.pcolormesh(X_2, T_2, rel_error_2, cmap = 'gist_ncar')
    axs1.set_xlabel(r'$x$')
    axs1.set_ylabel(r'$t$')
    cbar = fig1.colorbar(pan1, ax = axs1)
    cbar.set_label(r'Relative Error')

    plt.show()    



part_b()
