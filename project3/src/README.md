# Overview of files

## source files
The following files contains code used by the run files

 - `nn_eigval.py`: Tensorflow 1 implementation of neural network (NN) that solves for eigenvalues
 - `ode_model.py`: NN that solves an example ODE
 - `pde_model.py`: NN that solves the heat eq. NOT IN USE.
 - `pde_model_hm.py`: NN that solves the heat eq. IN USE.
 - `pde_solver.py`: Implementation of the forward euler algorithm to solve heat equation.

## run files
The following files can be executed to solve the project

  - `main_nn_eigval.py`: Executes eigenvalue NN and solves for largest and smallest eigenvalue.
  - `main_pde.py`: Executes forward euler solution to heat equation. 
  - `nn_eigval_hm.py`: Short implementation of the SGD theory in the paper that solves for eigenvalues without NN.
  - `nn_ode.py`: Executes NN that solves an example ODE.
  - `nn_pde.py`: Executes NN that solves heat equation.
  
  ## NOTE
  
  The eigenvalue neural network solver is somewhat unstable and can on occasion return an eigenvalue (and vector) smaller than the largest.
  Furthermore, it will often return -1* v, where v is the eigenvector found by numpy. It will therefore often in the plot appear that the eigenvector is wrong.
  It will most often simply be the sign that is wrong. Rerunning the code a couple of times will yield plots that "look" correct. 

