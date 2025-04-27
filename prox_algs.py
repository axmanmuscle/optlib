"""
Write some actual optimization algorithms out to use.
"""

import numpy as np
import prox

def dr_aoi(x0, proxf, proxg, t, maxiter=100, eps=1e-4):
    """
    douglas rachford splitting as an averaged operator iteration
    """

    Rf = lambda xin, gamma: 2*proxf(xin, gamma) - xin
    Rg = lambda xin, gamma: 2*proxg(xin, gamma) - xin

    S = lambda xin, gamma: Rg(Rf(xin, gamma), gamma)

    xi = x0
    for iter in range(maxiter):
        Sxi = S(xi, t)
        xi = 0.5*xi + 0.5*Sxi

        if np.linalg.norm(Sxi) < eps:
            break

    return xi

def douglasRachford(x0, proxf, proxg, t, maxiter=100, eps=1e-4):
    """
    basic version of Douglas Rachford splitting to solve
    min  f(x) + g(x)
     x
    """

    yi = np.zeros(x0.shape)


    for iter in range(maxiter):
        xi = proxf(yi, t)
        wi = proxg(2*xi - yi, t)
        yi = yi + wi - xi

        if np.linalg.norm(wi - xi) < eps:
            break
    
    return yi

def pdhg(x0, proxf, proxgconj, A, tau, sigma, alpha = 1, maxiter=100, eps=1e-4):
    """
    chambolle and pock's primal dual hybrid gradient method to solve problems
    min f(x) + g(Ax)
     x
    
    where f, g are proxable but nonsmooth and A is a linear operator.

    INPUTS:
        x0 - initial guess
        proxf - proximal operator for f
        proxgconj - proximal operator for g^*
        A - linear operator
            needs to accept a second argument that is 'transp' to indicate that it is the adjoint
            TODO: add ability to use a matrix as well
        tau - step size for f
        sigma - step size for g^*
        alpha - relaxation parameter (usually in (0, 2))
            TODO: add ability to use a sequence of alphas
        maxiter - maximum number of iterations
        eps - tolerance

    OUTPUTS:
        x - the optimal solution
        optional: the history of the residuals

    TODO:
        - check that norm A, sigma, tau satisfy the necessary conditions
        - add ability to use a matrix for A
        - add ability to use a sequence for alpha
        - add checking the relative change at each step to determine convergence/early stopping
    """

    xi = x0
    zi = A(xi)

    for iter in range(maxiter):
        xbar = proxf( xi - tau * A(zi, 'transp'), tau)
        zbar = proxgconj( zi + sigma * A(2*xbar - xi), sigma)
        xi = xi + alpha * (xbar - xi)
        zi = zi + alpha * (zbar - zi)

    return xi  

def pdhg_wls(x0, proxf, proxgconj, A, tau0, mu = 0.8, beta = 1, delta = 0.95, maxiter = 100):
    """
    implements malitsky and pock's line search for the primal dual hybrid gradient method
    """

    xi = x0
    zi = A(xi)

    taui = tau0
    thetai = 1
    for iter in range(maxiter):
        xold = xi
        zold = zi

        xi = proxf(xi - taui * A(zi, 'transp'), taui)
        tauold = taui
        taui = tauold*np.sqrt(1 + thetai) # this is our choice, we can choose any taui in [taui, taui*sqrt(1 + thetai)]
        accept = False
        ## line search
        while not accept:
            thetai = taui / tauold
            xbar = xi + thetai * (xi - xold)
            
            zi = proxgconj(zi + beta*taui*A(xbar), beta*taui)
            
            # compute condition for line search
            lside = np.sqrt(beta)*taui*np.linalg.norm(A(zi, 'transp') - A(zold, 'transp'))
            rside = delta * np.linalg.norm(zi - zold)
            if lside < rside:
                accept = True
            else:
                taui = mu*taui

    return xi

def main():
    return 0

if __name__ == "__main__":
    main()