"""The program runs the simulations and generates the plots needed for
the report Project 3
"""
import sys
sys.path.insert(0, "./src")


if __name__ == "__main__":
    import main_pde
    import nn_pde

    # Solves the PDE numerically
    main_pde.pde_numeric()
    # Solves the PDE by a neural network
    nn_pde.main(10)
    nn_pde.main(100)

    import main_nn_eigval

    # Solves the eigenvalue problem
    main_nn_eigval.main(-50)
