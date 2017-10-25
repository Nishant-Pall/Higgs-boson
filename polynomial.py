import numpy as np

def build_poly(x, degree):
    degree += 1
    
    basis = np.ones((x.shape[0], 1))
    for power in range(1, degree+1):
        newCol = np.power(x, power)
        basis = np.concatenate((basis, newCol), axis=1)

    return basis
