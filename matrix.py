
# matrix multiplication code from pset1
import numpy as np
from copy import deepcopy
from time import time


def matrix_mult(a, b):
    n = len(a)
    result = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return(result)


def add(a, b):
    n = len(a)
    result = deepcopy(a)
    for i in range(n):
        for j in range(n):
            result[i][j] += b[i][j]
    return(result)


def subtract(a, b):
    n = len(a)
    result = deepcopy(a)
    for i in range(n):
        for j in range(n):
            result[i][j] -= b[i][j]
    return(result)


def strassen_mult(a, b):
    n = len(a)

    # base case
    if n < 996:
        return(matrix_mult(a, b))
    # even case

    # define LETTERS
    A = a[: int(n/2), : int(n/2)]
    B = a[: int(n/2), int(n/2):]
    C = a[int(n/2):, : int(n/2)]
    D = a[int(n/2):,  int(n/2):]

    E = b[: int(n/2), : int(n/2)]
    F = b[: int(n/2), int(n/2):]
    G = b[int(n/2):, : int(n/2)]
    H = b[int(n/2):,  int(n/2):]

    p_1 = strassen_mult(A, subtract(F, H))
    p_2 = strassen_mult(add(A, B), H)
    p_3 = strassen_mult(add(C, D), E)
    p_4 = strassen_mult(D, subtract(G, E))
    p_5 = strassen_mult(add(A, D), add(E, H))
    p_6 = strassen_mult(subtract(B, D), add(G, H))
    p_7 = strassen_mult(subtract(A, C), add(E, F))
    # print(p_1, p_2, p_3, p_4, p_5, p_6, p_7)

    # define q's
    q_1 = add(subtract(add(p_5, p_4), p_2), p_6)
    q_2 = add(p_1, p_2)
    q_3 = add(p_3, p_4)
    q_4 = subtract(subtract(add(p_5, p_1), p_3), p_7)

    # Allocate zeros to C
    result = np.zeros((n, n)).astype('int')

    # Replace C
    result[0:int(n/2), 0:int(n/2)] = deepcopy(q_1)
    result[0:int(n/2), int(n/2):] = deepcopy(q_2)
    result[int(n/2):, 0:int(n/2)] = deepcopy(q_3)
    result[int(n/2):, int(n/2):] = deepcopy(q_4)

    return(result)


x = np.array([[2, 3, 4, 5],
              [2, 2, 6, 7],
              [3, 2, 4, 8],
              [1, 2, 3, 4]])

y = np.array([[2, 5, 7, 5],
              [3, 1, 12, 1],
              [1, 3, 2, 4],
              [4, 5, 6, 9]])

# x = np.array([[2, 3],
#               [2, 2]])
# y = np.array([[2, 5],
#               [3, 1]])

# x = np.array([[2, 3, 4],
#               [2, 2, 6],
#               [3, 2, 4]])
# y = np.array([[2, 5, 7],
#               [3, 1, 12],
#               [1, 3, 2]])


z = matrix_mult(
    [[2, 3, 4],
     [2, 2, 6],
     [3, 2, 4]],

    [[2, 5, 7],
     [3, 1, 12],
     [1, 3, 2]]
)

# print(y)

# q_1 = np.array([[1,2], [3,4]])
# q_2 = np.array([[5,6],[7,8]])
# q_3 = np.array([[1,5], [8,9]])
# q_4 = np.array([[12,6],[1,2]])

pow = 3
a = np.random.randint(0, 2, (2**pow, 2**pow))
b = np.random.randint(0, 2, (2**pow, 2**pow))

start = time()
strassen_mult(a, b)
end = time()
print("Strassen's: ", end="")
print(end-start)

start_2 = time()
matrix_mult(a, b)
end_2 = time()
print("Traditional: ", end="")
print(end_2 - start_2)


graph = np.random.binomial(1, .02, (2**5, 2**5))
adj = graph + graph.T - np.diag(graph.diagonal())
print(adj)

cubed_adj = strassen_mult(adj, strassen_mult(adj, adj))

print(cubed_adj)
triangles = 0
for i in range(len(adj)):
    triangles += cubed_adj[i][i]

triangles /= 6
print(triangles)

# print(matrix_mult(x, y))
# print(np.dot(a, b))
