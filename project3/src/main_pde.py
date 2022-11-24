import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import pde_solver


def part_b():

    #-------------------------------------------------------------------------------#
    # Uses the forward euler method in pde_solver.py
    # To solve the 1D diffusion equation with D = 1
    # Plots the solution as 2d colour gradients for 2 different values of x and t
    # Compares with analytical results and plots square error
    #
    # NB:
    # Figures are rasterized to save time/space when saving them
    #-------------------------------------------------------------------------------#

    # Set initial conditions
    dx_1 = 1/10
    dx_2 = 1/100

    Tmax_1 = 0.5
    Tmax_2 = 0.5

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
    sqr_error_1 = np.abs(U_anlytic_1 - u_1)**2 
    meansqrEr_1 = np.mean(sqr_error_1.flatten())
    mse_1 = np.zeros(len(t_1))
    for i in range(len(mse_1)):
        mse_1[i] = np.mean(sqr_error_1[i, :])

    sqr_error_2 = np.abs(U_anlytic_2 - u_2)**2 
    meansqrEr_2 = np.mean(sqr_error_2.flatten())
    mse_2 = np.zeros(len(t_2))
    for i in range(len(mse_2)):
        mse_2[i] = np.mean(sqr_error_2[i, :])

    # Set up figures

    fig1, axs1 = plt.subplots(figsize = (8,8))
    pan1 = axs1.pcolormesh(X_1, T_1, u_1, cmap = 'jet', rasterized=True)
    axs1.set_xlabel(r'$x$')
    axs1.set_ylabel(r'$t$')
    cbar = fig1.colorbar(pan1, ax = axs1)
    cbar.set_label(r'$u(x, t)$')
    #plt.savefig('u10.pdf', bbox_inches = 'tight')

    fig2, axs2 = plt.subplots(1, 2, figsize = (12, 8))
    pan2 = axs2[0].pcolormesh(X_1, T_1, sqr_error_1, cmap = 'gist_ncar', rasterized=True)
    axs2[0].set_xlabel(r'$x$')
    axs2[0].set_ylabel(r'$t$')
    cbar = fig2.colorbar(pan2, ax = axs2[0])
    cbar.set_label(r'Square Error')
    axs2[1].plot(t_1, mse_1, lw = 2, alpha = 0.7, color = 'r')
    axs2[1].yaxis.set_label_position('right')
    axs2[1].yaxis.tick_right()
    axs2[1].set_xlabel(r'timestep')
    axs2[1].set_ylabel(r'Mean Square Error')
    axs2[1].tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    axs2[1].tick_params(axis='x', top=True, direction = 'in', length = 10)
    axs2[1].tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    axs2[1].tick_params(axis='y', right=True, direction = 'in', length = 10)
    #plt.savefig('err10.pdf', bbox_inches = 'tight')

    fig3, axs3 = plt.subplots(figsize = (8,8))
    pan3 = axs3.pcolormesh(X_2, T_2, u_2, cmap = 'jet', rasterized=True)
    axs3.set_xlabel(r'$x$')
    axs3.set_ylabel(r'$t$')
    cbar = fig3.colorbar(pan3, ax = axs3)
    cbar.set_label(r'$u(x, t)$')
    #plt.savefig('u100.pdf', bbox_inches = 'tight')

    fig4, axs4 = plt.subplots(1, 2, figsize = (12, 8))
    pan4 = axs4[0].pcolormesh(X_2, T_2, sqr_error_2, cmap = 'gist_ncar', rasterized=True)
    axs4[0].set_xlabel(r'$x$')
    axs4[0].set_ylabel(r'$t$')
    cbar = fig4.colorbar(pan4, ax = axs4[0])
    cbar.set_label(r'Square Error')
    axs4[1].plot(t_2, mse_2, lw = 2, alpha = 0.7, color = 'r')
    axs4[1].yaxis.set_label_position('right')
    axs4[1].yaxis.tick_right()
    axs4[1].set_xlabel(r'timestep')
    axs4[1].set_ylabel(r'Mean Square Error')
    axs4[1].tick_params(axis='x', which='minor', top=True, direction = 'in', length = 5)
    axs4[1].tick_params(axis='x', top=True, direction = 'in', length = 10)
    axs4[1].tick_params(axis='y', which='minor', right=True, direction = 'in', length = 5)
    axs4[1].tick_params(axis='y', right=True, direction = 'in', length = 10)
    #plt.savefig('err100.pdf', bbox_inches = 'tight')

    plt.show()    



part_b()
