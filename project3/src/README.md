# Overview of files

## Source files
The following files contains code used by the run files

 - `nn_eigval.py`: Tensorflow 1 implementation of neural network (NN) that solves for eigenvalues
 - `ode_model.py`: NN that solves an example ODE
 - `pde_model.py`: NN that solves the heat eq.
 - `pde_solver.py`: Implementation of the forward euler algorithm to solve heat equation.

## Run files
The following files can be executed to solve the project

 - `main_nn_eigval.py`: Executes eigenvalue NN and solves for largest and smallest eigenvalue.
 - `main_pde.py`: Executes forward euler solution to heat equation. 
 - `nn_eigval_hm.py`: Short implementation of the SGD theory in the paper that solves for eigenvalues without NN.
 - `nn_ode.py`: Executes NN that solves an example ODE.
 - `nn_pde.py`: Executes NN that solves heat equation.
