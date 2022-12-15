# Overview of files

## Source files
The following files contains code used by the run files

 - `pde_solver.py`: Implementation of the forward Euler algorithm to solve the 
    heat equation.
 - `pde_model.py`: Neural network that solves the heat equation.
 - `nn_eigval.py`: Tensorflow 1 implementation of neural network that solves 
    for eigenvalues.
 - `ode_model.py`: Neural network that solves an ODE. (Not used in report.)

## Run files
The following files can be executed to solve the project. The first three are called by `../main.py`

 - `main_pde.py`: Executes forward Euler solution to heat equation.
 - `nn_pde.py`: Executes neural network that solves heat equation.
 - `main_nn_eigval.py`: Executes eigenvalue NN and solves for largest and 
    smallest eigenvalue.
 - `nn_eigval_hm.py`: Short implementation of the SGD theory in the paper that 
    solves for eigenvalues without NN.
 - `nn_ode.py`: Executes neural network that solves an ODE. (Not used in 
    report.).
