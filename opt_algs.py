import numpy as np

def simplex(c, A, b):
    """
    i want to code up the simplex method for linear programming

    maximize c^Tx st Ax <= b, x >= 0
    """

    # Convert inputs to NumPy arrays
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    m, n = A.shape  # m constraints, n variables
    if len(b) != m or len(c) != n:
        raise ValueError("Dimension mismatch in inputs.")
    
    # Initialize basis: assume last m variables are basic (slack variables)
    basis = list(range(n - m, n))
    non_basis = list(range(n - m))
    
    # Main Simplex loop
    while True:
        # Step 1: Compute basic solution
        B = A[:, basis]  # Basic columns
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise ValueError("Degenerate or infeasible problem.")
        
        x_b = B_inv @ b  # Basic variable values
        if np.any(x_b < 0):
            raise ValueError("Infeasible solution encountered.")
        
        x = np.zeros(n)
        for i, idx in enumerate(basis):
            x[idx] = x_b[i]
        
        # Step 2: Compute reduced costs
        c_b = c[basis]  # Objective coefficients for basic variables
        pi = c_b @ B_inv  # Dual variables
        reduced_costs = c - pi @ A  # Reduced costs for all variables
        
        # Step 3: Check optimality
        if np.all(reduced_costs[non_basis] <= 0):
            # Optimal solution found
            z = c @ x
            return x, z
        
        # Step 4: Select entering variable
        entering = non_basis[np.argmax(reduced_costs[non_basis])]
        if reduced_costs[entering] <= 0:
            raise ValueError("Problem is unbounded.")
        
        # Step 5: Compute pivot direction
        y = B_inv @ A[:, entering]
        if np.all(y <= 0):
            raise ValueError("Problem is unbounded.")
        
        # Step 6: Minimum ratio test to select leaving variable
        ratios = np.array([x_b[i] / y[i] if y[i] > 0 else np.inf for i in range(m)])
        leaving_idx = np.argmin(ratios)
        if ratios[leaving_idx] == np.inf:
            raise ValueError("Problem is unbounded.")
        
        # Step 7: Update basis
        leaving = basis[leaving_idx]
        basis[leaving_idx] = entering
        non_basis.remove(entering)
        non_basis.append(leaving)
        non_basis.sort()

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

