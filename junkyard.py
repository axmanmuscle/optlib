"""
idk junkyard things
numerical and autodifferentiation?
"""

import numpy as np
import matplotlib.pyplot as plt

def forward_fd(f, x0, h):
    """
    finite differencing approximation of the derivative
    forward diff
    """

    fprime = (1/h) * (f(x0 + h) - f(x0))
    return fprime

def backward_fd(f, x0, h):
    """
    finite differencing approximation of the derivative
    backward diff
    """

    fprime = (1/h) * (f(x0) - f(x0 - h))
    return fprime

def center_fd(f, x0, h):
    """
    finite differencing approximation of the derivative
    centered diff
    """

    fprime = (1/(2*h)) * (f(x0 + h) - f(x0 - h))
    return fprime

def imag_fd(f, x0, h):
    """
    finite differencing approximation of the derivative
    forward diff
    """

    fprime = (1/h) * np.imag((f(x0 + 1j*h) - f(x0)))
    return fprime

def main():
    # f = lambda x: 0.5*x**2
    # f = lambda x: 0.5*np.sin(x)
    f = lambda x: 0.5 * x**4 - 0.8*x**3

    x0 = 3 
    truth = 2 * (x0)**3 - 2.4*(x0)**2
    ies = [i for i in np.arange(-20, 0, 0.25)]
    hs = [10**i for i in ies]
    ff = []
    bf = []
    cf = []
    imf = []
    for h in hs:
    # h = 0.05
        ffd = forward_fd(f, x0, h)
        bfd = backward_fd(f, x0, h)
        cfd = center_fd(f, x0, h)
        ifd = imag_fd(f, x0, h)

        ff.append(np.abs(ffd - truth))
        bf.append(np.abs(bfd - truth))
        cf.append(np.abs(cfd - truth))
        imf.append(np.abs(ifd - truth))


    print(f'forward diff: {ffd}')
    print(f'backward diff: {bfd}')
    print(f'centered diff: {cfd}')
    print(f'imag diff: {ifd}')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(ies, ff)
    ax.semilogy(ies, bf)
    ax.semilogy(ies, cf)
    ax.semilogy(ies, imf)
    ax.legend(['forward', 'backward', 'centered', 'imag'])
    # ax.legend(['forward', 'centered'])
    plt.show()
    return 0

if __name__ == "__main__":
    main()