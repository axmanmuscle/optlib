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
