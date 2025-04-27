"""
Writing some useful proximal operators out to be used for different optimization problems.
"""
import numpy as np

def proxL2(x, sigma):
    """
    the proximal operator of || x ||_2
    """
    nx = np.linalg.norm(x, 2)
    m = np.max([nx, sigma])
    out = (1 - sigma/m)*x
    
    return out

def proxL2Sq(x, sigma):
    return 0

def proxL1(x, sigma, t = 1, b = None):
    """
    computes 
    prox_(sigma f) (x)
    where f(x) = t * || x ||_1

    INPUTS:
        x - nx1 numpy array representing the input
        sigma - scaling for proximal operator
        t - scaling for the function (if needed), 1 by default
    """

    xshape = x.shape
    if len(xshape) > 1:
        assert xshape[1] == 1, 'gave proxL1 a two dimensional (or larger) array'
    if b is None:
        b = np.zeros(xshape)

    magX = np.abs(x - b)
    thresh = sigma * t

    scalingFactors = thresh / magX
    out = np.zeros(xshape)

    nonzeroIndices = magX > thresh

    out[nonzeroIndices] = x[nonzeroIndices] * (1 - scalingFactors[nonzeroIndices])

    out = out + b

    return out

def l1Norm(x):
    return np.linalg.norm(x, 1)

def l2Norm(x):
    return np.linalg.norm(x, 2)

def l2Normsq(x):
    return np.linalg.norm(x, 2)**2

def testProx():
    print('proxL1 test')

    x = np.array([1,4,5,6,7])

    out = proxL1(x, 1)
    checkOut = np.array([0, 3, 4, 5, 6])
    assert l2Normsq(checkOut - out) < 1e-8, 'initial proxL1 test failed'

    out = proxL1(x, 1, 1, np.array([0,0,0,0,0]))
    assert l2Normsq(checkOut - out) < 1e-8, 'argument proxL1 test failed'

    out = proxL1(x, 2, 1, np.array([0,0,0,0,0]))
    checkOut = np.array([0, 2, 3, 4, 5])
    assert l2Normsq(checkOut - out) < 1e-8, 'sigma proxL1 test failed'

    out = proxL1(x, 1, 2, np.array([0,0,0,0,0]))
    assert l2Normsq(checkOut - out) < 1e-8, 't proxL1 test failed'

    out = proxL1(x, 0.5, 0.75, np.array([0,0,0,0,0]))
    checkOut = np.array([0.6250, 3.6250, 4.6250, 5.6250, 6.6250])
    assert l2Normsq(checkOut - out) < 1e-8, 't/sigma proxL1 test failed'

    out = proxL1(x, 0.5, 0.75, np.array([1,2,1,2,1]))
    checkOut = np.array([1, 5.25, 5.53125, 7.4375, 7.5625])
    assert l2Normsq(checkOut - out) < 1e-8, 'b proxL1 test failed'

    print('proxL1 passed')
    return 0

if __name__ == "__main__":
    testProx()