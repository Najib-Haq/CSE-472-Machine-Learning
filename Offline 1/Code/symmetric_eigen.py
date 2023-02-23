import numpy as np

def random_symmetric_invertible_matrix(n):
    """Return a random symmetric invertible matrix of size n x n."""

    # brute force
    while True:
        m = np.random.randint(-100, 100, size=(n, n))
        A = np.dot(m, m.T) # https://math.stackexchange.com/questions/158219/is-a-matrix-multiplied-with-its-transpose-something-special
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
    # np.seterr(all='raise')
    print(">>Input dimension n of matrix: ")
    n = int(input())
    
    # generate matrix
    A = random_symmetric_invertible_matrix(n)
    # A = np.array([[5329, -1666, 594, -6240], 
    #               [-1666, 3364, -702, -4410],
    #               [594,   -702, 2401, 117],
    #               [-6240, -4410, 117, 961]])
    print(">>Generated Matrix:\n", A)

    # eigen decomposition
    new_A = eigen_decomposition(A)

    # compare original and new matrix
    print("Reconstructed matrix is equal to original matrix: ", np.allclose(A, new_A))
    




