import numpy as np
import warnings
warnings.filterwarnings("ignore")

def random_matrix(n, m):
    return np.random.randint(-100, 100, size=(n, m))
    

def singular_value_decomposition(A):
    U, D, V_transposed = np.linalg.svd(A, full_matrices=True)
    # print(U, D, V_transposed)
    # D = np.zeros(shape=(U.shape[0], V_transposed.shape[0])) + np.diag(D)
    print(V_transposed.shape, U.shape)
    D = np.pad(np.diag(D), [(0, U.shape[0] - len(D)), (0, V_transposed.shape[0] - len(D))])
    # print(D)
    return U, D, V_transposed


def moore_penrose_pseudoinverse(U, D, V):
    D_plus = np.where(D > 0, 1/D, 0).T # take reciprocal of non-zero values and transpose
    # print(V.shape, D.shape, D_plus.shape, U.T.shape)
    return np.dot(V, np.dot(D_plus, U.T))


if __name__ == "__main__":
    print(">>Input dimension n & m of matrix: ")
    n, m = map(int, input().split())
    
    # generate matrix
    A = random_matrix(n, m)
    print(">>Generated Matrix: ", A)

    # SVD
    U, D, V_transposed = singular_value_decomposition(A)

    # Moore-Penrose pseudo inverse -> numpy
    A_inv = np.linalg.pinv(A)
    print("From numpy: ", A_inv)
    
    # Moore-Penrose pseudo inverse -> eq 2.47
    A_inv_2 = moore_penrose_pseudoinverse(U, D, V_transposed.T)
    print("From eq 2.47: ", A_inv_2)

    # compare the two inverses
    print("Compare numpy and eq 2.47: ", np.allclose(A_inv, A_inv_2))
    




