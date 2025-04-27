import numpy as np

def fista(x0, gradf, proxg, maxIter=100, objFun = None):
    """
    implementation of FISTA

    solves the problem
    min f(x) + g(x)
     x
    where f is smooth and g has a tractable proximal operator

    INPUTS:
        x0 - starting guess
        gradf - the gradient of f. should be a function handle
        proxg - the proximal operator of g. should be a function handle that tkes in
                the current x and a scaling factor
        (optional) maxIter - maximum number of iterations
        (optional) optFun - the objective function for objective values 
    OUTPUTS:
        xstar - the solution to the problem
    """

    return 0

def prox_grad(x0, gradf, proxg, maxIter=100, objFun = None, gamma=1):
    """
    let's implement the proximal gradient descent algorithm

    solves the problem
    min f(x) + g(x)
     x
    where f is smooth and g has a tractable proximal operator

    INPUTS:
        x0 - starting guess
        gradf - the gradient of f. should be a function handle
        proxg - the proximal operator of g. should be a function handle that tkes in
                the current x and a scaling factor
        (optional) maxIter - maximum number of iterations
        (optional) optFun - the objective function for objective values 
        (optional) gamma - the scaling for the proximal operator (default 1)
    OUTPUTS:
        xstar - the solution to the problem
    """

    # check whether the user passed in the objective function
    # this is necessary for the line search
    if callable(objFun):
        lineSearch = True
    else:
        lineSearch = False
        print("warning (prox_grad): no objective function provided. no line search will be used.")
        print("Not implemented yet")
        return -1

    if lineSearch:
        objVals = []
        obj0 = objFun(x0)
        objVals.append(obj0)

        # line search params

        rho = 0.9
        c = 0.9 # linesearch param
        tau = 1.1
        max_linesearch_iter = 100
        alpha = 1e-2

    xi = x0
    tol = 1e-6 # relative tolerance for convergence
    verbose = True # maybe add as an argument

    for idx in range(maxIter):
        # first the gradient descent step

        xold = xi

        if lineSearch:
            objVal = objFun(xi)
            linesearch_iter = 0
            obj_x = objFun(xi)
            ggrad_xi = gradf(xi)
            while linesearch_iter < max_linesearch_iter:
                linesearch_iter += 1
                xNew = xi - alpha*ggrad_xi
                obj_xnew = objFun(xNew)
                if obj_xnew < obj_x - alpha * c * np.linalg.norm(ggrad_xi.reshape(-1, 1))**2:
                    alpha_used = alpha
                    if verbose:
                        print(f'prox_grad: linesearch converged at iter {linesearch_iter} with alpha {alpha_used}')
                    break
                alpha *= rho

            if linesearch_iter == max_linesearch_iter:
                print(f'prox_grad: linesearch did not converge in {max_linesearch_iter} iterations, alpha {alpha}')
            xi = xNew
            alpha *= tau 

        else:
            alpha = 7e-6
            xi = xi - alpha * gradf(xi)
            alpha_used = alpha

        # now the proximal step
        xi = proxg(xi, alpha)

        tstr = f'prox_grad: iter {idx} '
        if lineSearch:
            obj1 = objFun(xi)
            tstr += f'obj {obj1} '
            objVals.append(obj1)

        if verbose:
            print(tstr)

        if np.linalg.norm(xi - xold) < tol * np.linalg.norm(xold):
            print(f'prox_grad: converged in {idx+1} iterations')
            break

    print(f'prox_grad: did not converge in {idx+1} iterations')
    # return final value of x
    return xi, objVals

