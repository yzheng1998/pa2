
# matrix multiplication code from pset1
import numpy as np


def matrix_mult(a, b):
    n = len(a)
    result = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return(result)


def add(a, b):


def strassen_mult(a, b):
    n = len(a)
    # even case


y = matrix_mult(
    [[2, 3, 4],
     [2, 2, 6],
     [3, 2, 4]],

    [[2, 5, 7],
     [3, 1, 12],
     [1, 3, 2]]
)

print(y)
