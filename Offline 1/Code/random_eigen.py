import numpy as np

def random_invertible_matrix(n):
    """Return a random invertible matrix of size n x n."""
    # https://stackoverflow.com/questions/73426718/generating-invertible-matrices-in-numpy-tensorflow
    # assumes has zero measure? 
    # A = np.random.randint(100, size=(n, n))
    # ax = np.sum(np.abs(A), axis=1) # generate summation of absolute values of each row -> if diag value > other values in row then invertible
    # np.fill_diagonal(A, ax)
    # return A
    
    # brute force
    while True:
        A = np.random.randint(-100, 100, size=(n, n))
        if np.linalg.det(A) != 0:
            return A
    

def eigen_decomposition(A):
    eigen_values, eigen_vectors = np.linalg.eig(A) #  column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    print(">>Eigen Values:\n", eigen_values)
    print(">>Eigen Vectors:\n", eigen_vectors)

    # reconstruct array from eigen values and eigen vectors
    diag_lambda = np.diag(eigen_values)
    new_A = np.dot(eigen_vectors, np.dot(diag_lambda, np.linalg.inv(eigen_vectors)))
    print(">>Reconstructed A:\n", new_A)
    return new_A


if __name__ == "__main__":
    print(">>Input dimension n of matrix: ")
    n = int(input())
    
    # generate matrix
    A = random_invertible_matrix(n)
    print(">>Generated Matrix: ", A)

    # eigen decomposition
    new_A = eigen_decomposition(A)

    # compare original and new matrix
    print("Reconstructed matrix is equal to original matrix: ", np.allclose(A, new_A))
    




