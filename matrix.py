
# matrix multiplication code from pset1
import numpy as np
from copy import deepcopy 


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
    result = a 
    for i in range(n): 
        for j in range(n): 
            result[i][j] += b[i][j]
    return(result)

def subtract(a, b): 
    n = len(a)
    result = a 
    for i in range(n): 
        for j in range(n): 
            result[i][j] -= b[i][j]
    return(result)

def strassen_mult(a, b):
    n = len(a)

    # even case

    # define LETTERS
    A = a[: int(n/2), : int(n/2)]
    B = a[ : int(n/2) , int(n/2) : ]
    C = a[int(n/2) :, : int(n/2)]
    D = a[int(n/2) : ,  int(n/2) : ]
    
    E = b[: int(n/2), : int(n/2)]
    F = b[ : int(n/2) , int(n/2) : ]
    G = b[int(n/2) :, : int(n/2)]
    H = b[int(n/2) : ,  int(n/2) : ]

    p_1 = strassen_mult(A, subtract(F, H))
    p_2 = strassen_mult(add(A, B), H)
    p_3 = strassen_mult(add(C, D), E)
    p_4 = strassen_mult(subtract(G, E), D)
    p_5 = strassen_mult(add(A, D), add(E, H))
    p_6 = strassen_mult(subtract(B, D), add(G, H))
    p_7 = strassen_mult(subtract(A, C), add(E, F))

    

    


print(strassen_mult(
    np.array([[2, 3, 4, 5],
     [2, 2, 6, 7],
     [3, 2, 4, 8], 
     [1, 2, 3, 4]]),

    np.array([[2, 5, 7, 5],
     [3, 1, 12, 1],
     [1, 3, 2, 4], 
     [4, 5, 6, 9]])
))

y = matrix_mult(
    [[2, 3, 4],
     [2, 2, 6],
     [3, 2, 4]],

    [[2, 5, 7],
     [3, 1, 12],
     [1, 3, 2]]
)

# print(y)
