import numpy as np 
import matplotlib.pyplot as plt

def initialCondition(x, L=1.):
    """
    Returns the initial condition at x
    """
    return np.sin((np.pi * x / L))

def analyticalSolution(x, t):
    """
    Returns analytical solution of diffusion equation
    """
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def forwardEuler(dx, T, L = 1., D = 1.):
    """
    Solves the diffusion equation

        u_t = D u_xx

    on x in [0, L] with the forward euler method.

    Boundary conditions: u(0, t) = u(L, t) = 0 for all t in T
    Initial condition:   u(x, 0) = sin(pi * x)

    Input
    -------------------
    dx : float          - The spacing in x
    T : int             - The total number of timesteps
    L : int             - The length of the rod
    D : float           - The diffusion constant

    Returns
    --------------------

    u : matrix          - The solution of the diffusion eq at all timestep
    x : array           - x-axis
    t : array           - t-axis
    """

    # 1. Set up x array. 
    Nx = int(L/dx)         # Number of points determined by length and spacing
    print(L, dx, Nx)
    x = np.linspace(0., L, Nx + 1)

    # 2. Set up t array
    # dt determined by  D * dt / (dx)**2 <= 1/2
    dt = dx**2 / (2*D)
    Nt = int(T/dt)
    print(dt, Nt)
    t = np.linspace(0, T, Nt + 1)

    a = dt/(dx)**2

    # 3. Set up containers for solutions at current and previous timesteps
    u = np.zeros((len(t), len(x)))

    # 4. Set u_1 to intial cond. and ensure boundary cond. is met
    u[0, :] = initialCondition(x)
    # u[0, 0] = 0.
    # u[0, Nx] = 0.

    # 5. Loop through timesteps and update u
    for i in range(Nt):
        #Update u according to theory, excluding 0 endpoints
        u[i+1, 1:Nx] = (1 - 2*a) * u[i, 1:Nx] + a*(u[i, :Nx-1] + u[i, 2:])

    return u, x, t

# Error functions
def sqr_err(f, g):
    return (f-g)**2

def mean_sqr_err(f, g):
    return np.mean(sqr_err(f,g))

def rel_err(f, g):
    return np.absolute((f-g)/g)