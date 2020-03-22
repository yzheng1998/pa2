
# matrix multiplication code from pset1
def matrix_mult(a, b): 
    result = [[0,0],[0,0]]
    for i in range(2): 
        for j in range(2): 
            for k in range(2): 
                result[i][j] += a[i][k] * b[k][j] 
    
    for i in range(2): 
        for j in range(2): 
            a[i][j] = result[i][j]
    return(a)

def matrix_mult_mod(a, b): 
    result = [[0,0],[0,0]]
    for i in range(2): 
        for j in range(2): 
            for k in range(2): 
                result[i][j] += a[i][k] * b[k][j] % 65536
    
    for i in range(2): 
        for j in range(2): 
            a[i][j] = result[i][j]
    return(a)