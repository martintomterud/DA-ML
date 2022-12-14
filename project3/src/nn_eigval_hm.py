"""Method to calculate eigenvector corresponding to the biggest eigenvalue,
for a symmetric matrix
"""
import numpy as np


def random_symmetric_matrix(matrix_size, rng=np.random.default_rng()):
    """Creates a random square matrix of specified size"""
    matrix = rng.standard_normal((matrix_size, matrix_size))
    return .5* (matrix + matrix.T)


def attempt(matrix_size, n, lr=1e-3, rng=np.random.default_rng()):
    """Attemt at eigensolver"""
    matrix = random_symmetric_matrix(matrix_size, rng)
    weights = rng.standard_normal(size=(matrix_size, 1))
    for _ in range(n):
        scalar1 = weights.T @ weights
        vec1 = matrix @ weights
        prod3 = (weights.T @ matrix @ weights)
        gradient = scalar1 * vec1 - prod3 * weights

        weights += lr * gradient

    eigenvals, eigenvecs = np.linalg.eigh(matrix)

    print("Norm of last gradient:")
    print(np.linalg.norm(gradient))

    print("Calculated eigenvector:")
    print(weights/np.linalg.norm(weights))
    print("Numpy eigenvectors:")
    print(eigenvecs[:,-1])
    print(np.linalg.norm(eigenvecs[:,-1]))



def main():
    """Testing eigensolving"""
    rng = np.random.default_rng()
    matrix_size = 6
    n = 10000
    lr = 1e-3
    attempt(matrix_size, n, lr, rng)

    return 0

if __name__ == "__main__":
    main()
