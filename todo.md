# MRI Utils
This is a collection of MRI reconstruction algorithms and various utilities to support it.


## To Do
### SENSE:
 - test adjoint of E
   - figure out why it's named E maybe change the name, too
 - implement gradient
 - try gradient descent on a modestly undersampled one
 - test actual (PISCO) vs. estimated (Dwork's code) sense maps

Okay what the fuck is going on though

### CS
 - Implement compressed sensing reconstruction

### Optimization Algorithms


## 314
Now the masking is broken, fix asap

### 319
Debug inverse wavelet transformation

## 4/3
Seems like sense is working now. This needs to become both the MRI utils and the math utils library with
 - opt algorithms
 - recon algorithms
 - prox/other operators
idk what the best way to organize that is though
oh well just write the damn code

change compressed sensing to solve for the wavelet coefficients instead

done. implement FISTA

## 4/27
turning this into a git repo. not sure the best way to organize it but want to add some optimization stuff

# From Other Project
### Optimization
This (I guess) will be optimization code that I actually end up writing in python (maybe other languages?). The idea is to have useful functions to run algorithms so I have a testbed to work on for stuff. I imagine having implementations of other algorithms and test problems that they use to solve.

### To Do
#### Algorithms
So far I mainly have proximal methods - Douglas Rachford. I need primal dual Douglas Rachford, PDHG, and Malitsky's PDHG line search. For other first order methods:
 - Gradient Descent
 - Proximal Gradient Descent (forward backward splitting)
 - My own generalized PDHG with line search
 - Condat's combination algorithm
 - ?

For second order methods:
 - Newton's
 - Proximal Newtons
 - Quasi Newton (BFGS?)
   - broyden
   - lazy newton
   - bfgs
   - l-bfgs
 - ?

Trust region methods:
 - interior point methods?

Other:
 - generalized lagrangian solving?

#### Other
I'd like some line searches. I know Hager and Zhang is a big one. Line searches for gradient descent as well?

Not sure what else. Put in on Github i guess.

#### Write test functions
Now that some of the algorithms have been written we need tests to make sure they're working correctly. Some simple problems just to double check the output of e.g. the prox operators against MATLAB implementations and make sure everything's working correctly.

