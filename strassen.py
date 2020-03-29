import sys
import random
from copy import deepcopy
from time import time
import fileinput


def matrix_mult(a, b):
    n = len(a)
    result = [[0 for y in range(n)] for x in range(n)]
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
    if n <= 2**7:
        return(matrix_mult(a, b))
    # even case

    # new define LETTERS
    A = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            A[i][j] = a[i][j]

    B = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            B[i][j] = a[i][j + int(n/2)]

    C = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            C[i][j] = a[i + int(n/2)][j]

    D = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            D[i][j] = a[i + int(n/2)][j + int(n/2)]

    E = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            E[i][j] = b[i][j]

    F = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            F[i][j] = b[i][j + int(n/2)]

    G = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            G[i][j] = b[i + int(n/2)][j]

    H = [[0 for y in range(int(n/2))] for x in range(int(n/2))]
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            H[i][j] = b[i + int(n/2)][j + int(n/2)]

    p_1 = strassen_mult(A, subtract(F, H))
    p_2 = strassen_mult(add(A, B), H)
    p_3 = strassen_mult(add(C, D), E)
    p_4 = strassen_mult(D, subtract(G, E))
    p_5 = strassen_mult(add(A, D), add(E, H))
    p_6 = strassen_mult(subtract(B, D), add(G, H))
    p_7 = strassen_mult(subtract(A, C), add(E, F))

    # define q's
    q_1 = add(subtract(add(p_5, p_4), p_2), p_6)
    q_2 = add(p_1, p_2)
    q_3 = add(p_3, p_4)
    q_4 = subtract(subtract(add(p_5, p_1), p_3), p_7)

    # Allocate zeros to C
    result = [[0 for y in range(n)] for x in range(n)]

    # Replace C
    for i in range(int(n/2)):
        for j in range(int(n/2)):
            result[i][j] = q_1[i][j]

    for i in range(int(n/2)):
        for j in range(int(n/2)):
            result[i][j + int(n/2)] = q_2[i][j]

    for i in range(int(n/2)):
        for j in range(int(n/2)):
            result[i + int(n/2)][j] = q_3[i][j]

    for i in range(int(n/2)):
        for j in range(int(n/2)):
            result[i + int(n/2)][j + int(n/2)] = q_4[i][j]
    return(result)


def randomMatrix(n):
    matrix = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = random.randint(-1, 1)
    return matrix


def test(n):
    a = randomMatrix(n)
    b = randomMatrix(n)
    print("Strassen's")
    print("Traditional")
    start = time()
    strassen = strassen_mult(a, b)
    end = time()
    strassen_time = end-start
    print(strassen_time)

    start_2 = time()
    traditional = matrix_mult(a, b)
    end_2 = time()
    traditional_time = end_2 - start_2
    print(traditional_time)

    print("Strassen's is faster:", strassen_time < traditional_time)
    print("Strassen's is correct:", strassen == traditional)


dim = int(sys.argv[2])

matrixArr = []
for line in fileinput.input(sys.argv[3]):
    matrixArr.extend([int(line)])


def createMatrices():
    a = [[0] * dim for i in range(dim)]
    b = [[0] * dim for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            a[i][j] = matrixArr.pop(0)
    for i in range(dim):
        for j in range(dim):
            b[i][j] = matrixArr.pop(0)
    return a, b


a, b = createMatrices()
product = strassen_mult(a, b)
for i in range(dim):
    print(product[i][i])

# pow = 8
# test(2**pow)


# Problem 3
V = 1024
p = .05


def generateGraph(prob):
    graph = [[0] * V for i in range(V)]
    for i in range(V):
        for j in range(V):
            if (i < j):
                graph[i][j] = int(random.random() < prob)
    for i in range(V):
        for j in range(V):
            if (i > j):
                graph[i][j] = deepcopy(graph[j][i])
    return graph


def trace(graph):
    trace = 0
    for i in range(V):
        trace += graph[i][i]
    return trace


def triangles(adj):
    cubed_adj = strassen_mult(
        deepcopy(adj), strassen_mult(deepcopy(adj), deepcopy(adj)))
    print(trace(cubed_adj))
    triangles = trace(cubed_adj)
    return triangles // 6


CONSTANT = 178433024


def expected():
    expectedArr = []
    for i in range(1, 6):
        prob = i / 100
        expectedArr.extend([CONSTANT*(prob**3)])
    return expectedArr


def experiment_prob(runs, prob):
    total = 0
    for i in range(runs):
        adj = generateGraph(prob)
        total += triangles(adj)
    ave = total / runs
    return ave


def experiment():
    results = []
    for i in range(1, 6):
        prob = i / 100
        print(prob)
        results.extend(experiment_prob(5, prob))
    return results


# print(expected())
# print(experiment())


# print("expected", expected(p))

# print(matrix_mult(x, y))
# print(np.dot(a, b))
