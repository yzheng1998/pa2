
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


y = matrix_mult(
    [[2, 3],
     [2, 2]],

    [[2, 5],
     [3, 1]]
)

print(y)
