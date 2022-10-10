# Overview of code files

User warning: One of the files (regressionClass.py) uses Python 3.10 match - case functionality. 
Use an updated python distribution when executing.

## Source codes

 - bootstrap.py : Class implementation of bootstrap resampling. No longer in use in final version.
 - designmMatrix.py : CLass implementation of design matrix. No longer in use in final version. 
 - franke.py : Functions that generate the franke function and a noisy version of the franke function.
 - regressionClass.py : Class implementation of the different regression intances, with functions that also compute predictions 
 - datafunctions.py : Different fuctions for loading terrain data, creating design matrix, performing bootstrap and k-fold cross validation and computing statistics
 
## Executing codes

When running the instances, please make sure to uncomment the function you want to run, 
and make not that we create a path using os('.') + '/figures/' to save figures. 
If this path does not exist in your environment, you might want to edit or comment this line to avoid bugs. 

Ridge and lasso uses the same function. Switch between 'ridge' and 'lasso' where method is specified to check the different regression methods.

- test.py : simple script to test that stuff runs
- testFrankeOLS.py : as above, simple script that test different code functions
- mainOLS.py : The main instance that performs OLS reg on the noisy Franke function and creates figures.
- mainRidgeLasso.py : Performs ridge and lasso reg on Franke function. 
- terrainMain.py : Contains all regression methods on the terrain data.
- terrainFigure.py : Creates the terrain figure in the report. 
