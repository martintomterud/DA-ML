"""Solving a PDE numerically"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pde_solver


#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
# plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
plt.rcParams.update({'figure.autolayout': True})
# from matplotlib.lines import Line2D
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator)



def part_b():
    """Uses the forward Euler method from pde_solver.py to solve the 1d
    diffusion equation with D = 1. Plots the solution as 2d colour gradients
    for 2 different values of x and t. Compares with analytical results and
    plots square error.
    NB: Figures are rasterized to save time/space when saving them
    """
    # Set initial conditions
    dx_1 = 1/10
    dx_2 = 1/100

    Tmax_1 = 0.5
    Tmax_2 = 0.5

    # Sol 1, 1
    u_1, x_1, t_1 = pde_solver.forward_euler(dx_1, Tmax_1)
    # Meshgrid and analytical solution
    X_1, T_1 = np.meshgrid(x_1, t_1)
    U_analytic_1 = pde_solver.analytical_solution(X_1, T_1)

    # Sol 2, 2
    u_2, x_2, t_2 = pde_solver.forward_euler(dx_2, Tmax_2)
    # Meshgrid and analytical solution
    X_2, T_2 = np.meshgrid(x_2, t_2)
    U_analytic_2 = pde_solver.analytical_solution(X_2, T_2)

    # Compute relative errors
    sqr_error_1 = pde_solver.sqr_err(U_analytic_1, u_1)
    rel_err_1 = pde_solver.rel_err(u_1[:, 1:-1], U_analytic_1[:, 1:-1])
    mse_1 = np.zeros(len(t_1))
    rel_1 = np.zeros(len(t_1))
    for i in range(len(t_1)):
        mse_1[i] = np.mean(sqr_error_1[i])
        rel_1[i] = np.mean(rel_err_1[i])

    sqr_error_2 = pde_solver.sqr_err(U_analytic_2, u_2)
    rel_err_2 = pde_solver.rel_err(U_analytic_2[:, 1:-1], u_2[:, 1:-1])
    mse_2 = np.zeros(len(t_2))
    rel_2 = np.zeros(len(t_2))
    for i in range(len(mse_2)):
        mse_2[i] = np.mean(sqr_error_2[i])
        rel_2[i] = np.mean(rel_err_2[i])

    # Set up figures

    fig1, axs1 = plt.subplots(figsize=(8,8))
    pan1 = axs1.pcolormesh(X_1, T_1, u_1, cmap='Reds', rasterized=True)
    axs1.set_xlabel(r'$x$')
    axs1.set_ylabel(r'$t$')
    cbar = fig1.colorbar(pan1, ax=axs1)
    cbar.set_label(r'$u(x, t)$')
    plt.savefig('num_u10.pdf', bbox_inches='tight')

    fig2, axs2 = plt.subplots(1, 2, figsize=(12,8))
    pan2 = axs2[0].pcolormesh(X_1, T_1, sqr_error_1, cmap='gist_ncar', rasterized=True)
    axs2[0].set_xlabel(r'$x$')
    axs2[0].set_ylabel(r'$t$')
    cbar = fig2.colorbar(pan2, ax=axs2[0])
    cbar.set_label(r'Squared error')
    axs2[1].plot(t_1, mse_1, lw=2, alpha=0.7, color='r')
    axs2[1].yaxis.set_label_position('right')
    axs2[1].yaxis.tick_right()
    axs2[1].set_xlabel(r'$t$')
    axs2[1].set_ylabel(r'Mean squared error')
    axs2[1].tick_params(axis='x', which='minor', top=True, direction='in', length=5)
    axs2[1].tick_params(axis='x', top=True, direction='in', length=10)
    axs2[1].tick_params(axis='y', which='minor', right=True, direction='in', length=5)
    axs2[1].tick_params(axis='y', right=True, direction='in', length=10)
    plt.savefig('num_err10.pdf', bbox_inches='tight')

    fig3, axs3 = plt.subplots(figsize=(8,8))
    pan3 = axs3.pcolormesh(X_2, T_2, u_2, cmap='Reds', rasterized=True)
    axs3.set_xlabel(r'$x$')
    axs3.set_ylabel(r'$t$')
    cbar = fig3.colorbar(pan3, ax=axs3)
    cbar.set_label(r'$u(x, t)$')
    plt.savefig('num_u100.pdf', bbox_inches='tight')

    fig4, axs4 = plt.subplots(1, 2, figsize=(12,8))
    pan4 = axs4[0].pcolormesh(X_2, T_2, sqr_error_2, cmap='binary', rasterized=True)
    axs4[0].set_xlabel(r'$x$')
    axs4[0].set_ylabel(r'$t$')
    cbar = fig4.colorbar(pan4, ax=axs4[0])
    cbar.set_label(r'Squared error')
    axs4[1].plot(t_2, mse_2, lw=2, alpha=0.7, color='k')
    axs4[1].yaxis.set_label_position('right')
    axs4[1].yaxis.tick_right()
    axs4[1].set_xlabel(r'$t$')
    axs4[1].set_ylabel(r'Mean squared error')
    axs4[1].tick_params(axis='x', which='minor', top=True, direction='in', length=5)
    axs4[1].tick_params(axis='x', top=True, direction='in', length=10)
    axs4[1].tick_params(axis='y', which='minor', right=True, direction='in', length=5)
    axs4[1].tick_params(axis='y', right=True, direction='in', length=10)
    plt.savefig('num_err100.pdf', bbox_inches='tight')

    # Plots of relative error
    fig5, axs5 = plt.subplots(1, 2, figsize=(12,8))
    pan5 = axs5[0].pcolormesh(
        X_1[:,1:-1],
        T_1[:,1:-1],
        rel_err_1,
        cmap='gist_ncar',
        rasterized=True
    )
    axs5[0].set_xlabel(r'$x$')
    axs5[0].set_ylabel(r'$t$')
    cbar = fig5.colorbar(pan5, ax=axs5[0])
    cbar.set_label(r'Relative error')
    axs5[1].plot(t_1, rel_1, lw=2, alpha=0.7, color='r')
    axs5[1].yaxis.set_label_position('right')
    axs5[1].yaxis.tick_right()
    axs5[1].set_xlabel(r'$t$')
    axs5[1].set_ylabel(r'Relative error')
    axs5[1].tick_params(axis='x', which='minor', top=True, direction='in', length=5)
    axs5[1].tick_params(axis='x', top=True, direction='in', length=10)
    axs5[1].tick_params(axis='y', which='minor', right=True, direction='in', length=5)
    axs5[1].tick_params(axis='y', right=True, direction='in', length=10)
    fig5.savefig("num_rel_err10")

    fig6, axs6 = plt.subplots(1, 2, figsize=(12,8))
    pan6 = axs6[0].pcolormesh(X_2[:,1:-1], T_2[:,1:-1], rel_err_2, cmap='Greys', rasterized=True)
    axs6[0].set_xlabel(r'$x$')
    axs6[0].set_ylabel(r'$t$')
    cbar = fig6.colorbar(pan6, ax=axs6[0])
    cbar.set_label(r'Relative error')
    axs6[1].plot(t_2, rel_2, lw=2, alpha=0.7, color='k')
    axs6[1].yaxis.set_label_position('right')
    axs6[1].yaxis.tick_right()
    axs6[1].set_xlabel(r'$t$')
    axs6[1].set_ylabel(r'Relative error')
    axs6[1].tick_params(axis='x', which='minor', top=True, direction='in', length=5)
    axs6[1].tick_params(axis='x', top=True, direction='in', length=10)
    axs6[1].tick_params(axis='y', which='minor', right=True, direction='in', length=5)
    axs6[1].tick_params(axis='y', right=True, direction='in', length=10)
    fig6.savefig("num_rel_err100.pdf")

    return 0

part_b()
