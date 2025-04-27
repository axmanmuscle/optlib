import numpy as np
from scipy.stats import laplace, norm
import matplotlib.pyplot as plt
import glob
import h5py
import scipy.io as sio

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
    # if len(xshape) > 1:
    #     assert xshape[1] == 1, 'gave proxL1 a two dimensional (or larger) array'
    if b is None:
        b = np.zeros(xshape)

    magX = np.abs(x - b)
    thresh = sigma * t

    scalingFactors = thresh / magX
    out = np.zeros(xshape, dtype=x.dtype)

    nonzeroIndices = magX > thresh

    out[nonzeroIndices] = x[nonzeroIndices] * (1 - scalingFactors[nonzeroIndices])

    out = out + b

    return out

def makeWavSplit(sImg, minSplitSize=16):
    """
    make the split to be used iwht the wavelet transforms
    INPUTS:
        sImg - size of the image as a tuple or list
        (OPTIONAL) minSplitSize - minimum size of split, default 16
    """

    def findNPows(sizeDim, minSplitSize):
        binPow = np.log2(sizeDim)

        nPow = 0
        for powIdx in range(np.floor(binPow).astype(int)):
            if sizeDim % 2 == 0 and sizeDim/2 >= minSplitSize:
                nPow += 1
                sizeDim = sizeDim/2
            else:
                break
        
        return nPow

    nDims = len(sImg)
    nPows = np.zeros(shape=(1, nDims))

    if np.size(minSplitSize) == 1:
        minSplitSize = minSplitSize * np.ones(shape=(nDims, 1))

    for dimIdx in range(nDims):
        nPows[0, dimIdx] = findNPows( sImg[dimIdx], minSplitSize[dimIdx, 0])

    if nDims == 1:
        wavSplit = np.zeros(shape=(2**(nPows-1), 1))
    else:
        wavSplit = np.zeros(*np.power(2, nPows-1).astype(int))

    wavSplit[0, 0] = 1

    sWavSplit = wavSplit.shape
    wavSplit = wavSplit[0:np.min(sWavSplit), 0:np.min(sWavSplit)]

    return wavSplit

def powerIter(A, x0, tol = 1e-4, maxIter=100):
    """
    power iteration for finding the largest eigenvalue of a function
    """

    x = x0
    lamb = 0

    verbose = True
    for i in range(maxIter):
        AtAx = A( A(x), 'transp')
        lastLamb = lamb
        lamb = np.linalg.norm(AtAx.flatten())
        if lamb == 0:
            break
        if verbose:
            print(f'Iteration {i}: Est = {np.sqrt(lamb)}')

        x = AtAx / lamb

        diff = np.abs(lamb - lastLamb) / lamb
        if diff < tol:
            if verbose:
                print(f'Converged after {i} iterations')
            break
    
    return np.sqrt(lamb)

def view_im_cube(data, title=''):
    """
    the expectation here is that we're plotting data of size [Nx x Ny x C]
    and we will make C subplots
    """
    sImg = data.shape
    fig = plt.figure()
    C = sImg[-1]
    cols = int(np.ceil(C/2))
    rows = int(np.ceil(C/cols))

    for i in range(C):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(np.abs(data[:,:,i]), cmap='gray')
        if len(title) == 0:
            ax.set_title('Coil ' + str(i))

    if len(title) > 0:
        plt.suptitle(title)

    plt.show()

def imspace_to_kspace(im_space):

    k_space = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( im_space, axes=(0, 1)), axes=(0, 1) ), axes=(0, 1))

    return k_space

def kspace_to_imspace(kspace):

    im_space = np.fft.fftshift( np.fft.ifftn( np.fft.ifftshift( kspace, axes=(0, 1)),  axes=(0, 1) ), axes=(0,1))

    return im_space

def view_im(kspace, title=''):

    im_space = kspace_to_imspace(kspace)

    plt.imshow( np.abs( im_space ), cmap='grey')

    if len(title) > 0:
        plt.title(title)
    plt.show(block=False)
    
def size2imgCoordinates(n):
    """
    INPUTS:
        n - array giving number of elements in each dimension
    OUTPUTS:
        coords - if n is a scalar, a 1D array of image coordinates
                 if n is an array, then a (size(n) x 1) array of image coordinates
    """
    if type(n) == list or tuple:
        numN = len(n)
        coords = []
        for i in range(numN):
            coords.append(size2img1d(n[i]))
    else:
        coords = [size2img1d(n)]

    return coords

def size2img1d(N):
    coords = np.array([i for i in range(N)]) - np.floor(0.5*N)
    return coords.astype('int')

def vdSampleMask(smask, sigmas, numSamps, maskType = 'laplace'):
    """
    generates a vd sample mask
    INPUTS:
        smask - 1-D array corresponding to number of samples in each dimension
        sigmas - 1-D array corresponding to the standard deviation of the distribution
                 in each dimension
        numSamps - number of (total) samples

    """
    maxIters = 500
    rng = np.random.default_rng(20230911)

    coords = size2imgCoordinates(smask)

    mask = np.zeros(smask)
    nDims = len(smask) # can't pass in just an integer

    if maskType == 'laplace':
        pdf = lambda x, sig: laplace.pdf(x, loc=0, scale=np.sqrt(0.5*sig*sig))
    elif maskType == 'gaussian':
        pdf = lambda x, sig: norm.pdf(x, loc=0, scale=sig)

    for idx in range(maxIters):
        sampsLeft = int(numSamps - mask.sum(dtype=int))
        dimSamps = np.zeros((nDims, sampsLeft))
        for dimIdx in range(nDims):
            c = coords[dimIdx]
            probs = pdf(c, sigmas[dimIdx])
            probs = probs / sum(probs)
            samps = rng.choice(c, sampsLeft, p=probs)
            dimSamps[dimIdx, :] = samps - min(c)
        
        mask[tuple(dimSamps.astype(int))] = 1

        if mask.sum(dtype=int) == numSamps:
            return mask
    
    print('hit max iters vdSampleMask')
    return mask

def upsample2(img, U):
    """
    only implementing the forward method for right now
    """

    sImg = img.shape
    ndims = len(sImg)
    S = np.zeros((ndims, 1), dtype=np.int64)

    sOut = np.array(sImg) * np.array(U) # this might be fragile

    yqs = S[0] + np.arange(0, sImg[0])*U[0]
    xqs = S[1] + np.arange(0, sImg[1])*U[1]

    out = np.zeros(sOut, dtype=img.dtype)
    out[np.ix_(yqs, xqs)] = img

    return out

def wtDaubechies2( img, split = np.array([1]) ):
    """
    applies the 2 dimensional Daubechies 4 wavelet transform
    INPUTS:
        img - 2d img
        split - (OPTIONAL) describes the way to split
    """
    sSplit = split.shape

    imgrt3 = np.sqrt(3) * img
    img3 = 3 * img

    imgPimgrt3 = img + imgrt3
    img3Pimgrt3 = img3 + imgrt3
    imgMimgrt3 = img - imgrt3
    img3Mimgrt3 = img3 - imgrt3

    wt1 = imgMimgrt3 + np.roll(img3Mimgrt3, -1, axis=0) \
                     + np.roll(img3Pimgrt3, -2, axis=0) \
                     + np.roll(imgPimgrt3, -3, axis=0)
    wt1 = wt1[::2, :]

    wt1rt3 = wt1 * np.sqrt(3)
    wt13 = wt1 * 3

    wt1Pwt1rt3 = wt1 + wt1rt3
    wt13Pwt1rt3 = wt13 + wt1rt3
    wt1Mwt1rt3 = wt1 - wt1rt3
    wt13Mwt1rt3 = wt13 - wt1rt3

    wt11 = wt1Mwt1rt3 + np.roll(wt13Mwt1rt3, [0, -1], axis=[0, 1]) \
                      + np.roll(wt13Pwt1rt3, [0, -2], axis=[0, 1]) \
                      + np.roll(wt1Pwt1rt3, [0, -3], axis=[0, 1])
    
    wt11 = wt11[:, ::2]

    wt12 = -1*wt1Pwt1rt3 + np.roll(wt13Pwt1rt3, [0, -1], axis=[0, 1]) \
                         + np.roll(-1*wt13Mwt1rt3, [0, -2], axis=[0, 1]) \
                         + np.roll(wt1Mwt1rt3, [0, -3], axis=[0, 1])
    wt12 = wt12[:, ::2]

    wt2 = -1*imgPimgrt3 + np.roll(img3Pimgrt3, [-1, 0], axis=[0, 1]) \
                        + np.roll(-1*img3Mimgrt3, [-2, 0], axis=[0, 1]) \
                        + np.roll(imgMimgrt3, [-3, 0], axis=[0, 1])
    wt2 = wt2[::2, :]

    wt2rt3 = wt2 * np.sqrt(3)
    wt23 = wt2 * 3

    wt2Pwt2rt3 = wt2 + wt2rt3
    wt23Pwt2rt3 = wt23 + wt2rt3
    wt2Mwt2rt3 = wt2 - wt2rt3
    wt23Mwt2rt3 = wt23 - wt2rt3

    wt21 = wt2Mwt2rt3 + np.roll(wt23Mwt2rt3, [0, -1], axis=[0, 1]) \
                      + np.roll(wt23Pwt2rt3, [0, -2], axis=[0, 1]) \
                      + np.roll(wt2Pwt2rt3, [0, -3], axis=[0, 1])
    wt21 = wt21[:, ::2]

    wt22 = -1*wt2Pwt2rt3 + np.roll(wt23Pwt2rt3, [0, -1], axis=[0, 1]) \
                         + np.roll(-1*wt23Mwt2rt3, [0, -2], axis=[0, 1]) \
                         + np.roll(wt2Mwt2rt3, [0, -3], axis=[0, 1])
    wt22 = wt22[:, ::2]

    nSplit = split.size
    if nSplit > 1:
        s11 = split[0:sSplit[0]//2,0:sSplit[1]//2]
        s12 = split[0:sSplit[0]//2, sSplit[1]//2+1:]
        s21 = split[sSplit[1]//2+1:, 0:sSplit[0]//2]
        s22 = split[sSplit[1]//2+1:, sSplit[1]//2+1:]

        if s11.sum() > 0:
            if np.any(np.mod(wt11.shape, 2)):
                raise ValueError('wt11 is invalid shape')
            wt11 = wtDaubechies2(wt11, s11)

        if s12.sum() > 0:
            if np.any(np.mod(wt12.shape, 2)):
                raise ValueError('wt12 is invalid shape')
            wt12 = wtDaubechies2(wt12, s12)

        if s21.sum() > 0:
            if np.any(np.mod(wt21.shape, 2)):
                raise ValueError('wt21 is invalid shape')
            wt21 = wtDaubechies2(wt21, s21)

        if s22.sum() > 0:
            if np.any(np.mod(wt22.shape, 2)):
                raise ValueError('wt22 is invalid shape')
            wt22 = wtDaubechies2(wt22, s22)


    a1 = np.concatenate([wt11, wt12], axis=1)
    a2 = np.concatenate([wt21, wt22], axis=1)
    wt = np.concatenate([a1, a2], axis=0)

    wt /= 32

    return wt

def iwtDaubechies2(wt, split = np.array([1])):
    """
    inverse Daubechies wavelet transformation
    """

    sWT = wt.shape
    ## TODO check that the sizes are divisible by two?
    wt11 = wt[:sWT[0]//2, :sWT[1]//2]
    wt21 = wt[sWT[0]//2:, :sWT[1]//2]
    wt12 = wt[:sWT[0]//2, sWT[1]//2:]
    wt22 = wt[sWT[0]//2:, sWT[1]//2:]

    sSplit = split.shape
    if np.max( np.mod( np.log2( sSplit), 1) ) > 0:
        raise ValueError('something in the split is the wrong size')
    nSplit = split.size
    if nSplit > 1:
        s11 = split[:sSplit[0]//2, :sSplit[1]//2]
        s12 = split[:sSplit[0]//2, sSplit[1]//2:]
        s21 = split[sSplit[1]//2:, :sSplit[0]//2]
        s22 = split[sSplit[1]//2:, sSplit[1]//2:]

        if s11.sum() > 0:
            if np.any(np.mod(wt11.shape, 2)):
                raise ValueError('wt11 is invalid shape')
            wt11 = iwtDaubechies2(wt11, s11)
        if s12.sum() > 0:
            if np.any(np.mod(wt12.shape, 2)):
                raise ValueError('wt12 is invalid shape')
            wt12 = iwtDaubechies2(wt12, s12)
        if s21.sum() > 0:
            if np.any(np.mod(wt21.shape, 2)):
                raise ValueError('wt21 is invalid shape')
            wt21 = iwtDaubechies2(wt21, s21)
        if s22.sum() > 0:
            if np.any(np.mod(wt22.shape, 2)):
                raise ValueError('wt22 is invalid shape')
            wt22 = iwtDaubechies2(wt22, s22)
    
    ## todo: write upsample
    tmp = upsample2(wt11, [1, 2])

    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt1_1 = tmp - tmprt3 + np.roll(tmp3 - tmprt3, [0, 1], axis = [0, 1]) \
                         + np.roll( tmp3 + tmprt3, [0, 2], axis=[0, 1]) \
                         + np.roll(tmp + tmprt3, [0, 3], axis = [0, 1])
    
    tmp = upsample2(wt12, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt1_2 = -1 * (tmp + tmprt3) + np.roll(tmp3 + tmprt3, [0, 1], axis = [0, 1]) \
                                + np.roll( -1 * (tmp3 - tmprt3), [0, 2], axis=[0, 1]) \
                                + np.roll(tmp - tmprt3, [0, 3], axis = [0, 1])
    
    wt1 = upsample2( wt1_1 + wt1_2, [2, 1])
    
    tmp = upsample2(wt21, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt2_1 = tmp - tmprt3 + np.roll(tmp3 - tmprt3, [0, 1], axis = [0, 1]) \
                         + np.roll(tmp3 + tmprt3, [0, 2], axis=[0, 1]) \
                         + np.roll(tmp + tmprt3, [0, 3], axis = [0, 1])
    
    tmp = upsample2( wt22, [1, 2])
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    wt2_2 = -1 * (tmp + tmprt3) + np.roll(tmp3 + tmprt3, [0, 1], axis = [0, 1]) \
                                + np.roll(-1 * (tmp3 - tmprt3), [0, 2], axis=[0, 1]) \
                                + np.roll(tmp - tmprt3, [0, 3], axis = [0, 1])
    
    wt2 = upsample2( wt2_1 + wt2_2, [2, 1])

    tmp = wt1
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    sig1 = tmp - tmprt3 + np.roll( tmp3 - tmprt3, [1, 0], axis = [0, 1]) \
                       + np.roll( tmp3 + tmprt3, [2, 0], axis = [0, 1]) \
                       + np.roll( tmp + tmprt3, [3, 0], axis = [0, 1])
    
    tmp = wt2
    tmp3 = 3 * tmp
    tmprt3 = np.sqrt(3) * tmp

    sig2 = -1*(tmp + tmprt3) + np.roll(tmp3 + tmprt3, [1, 0], axis = [0, 1]) \
                             + np.roll( -1*( tmp3 - tmprt3), [2, 0], axis = [0, 1]) \
                             + np.roll( tmp - tmprt3, [3, 0], axis = [0, 1])
    
    img = (sig1 + sig2) / 32

    return img

def mri_reconRoemer(coilRecons, sMaps = []):
    """
    implements the Roemer reconstruction method
    https://pubmed.ncbi.nlm.nih.gov/2266841/
    numpy version
    """

    if len(sMaps) == 0:
        ssqRecon = np.sqrt( np.sum( coilRecons * np.conj(coilRecons), 2 ))
        sMaps = coilRecons / ssqRecon
        sMaps[sMaps == np.inf] = 0
    
    recon = np.sum( coilRecons * np.conj(sMaps), 2)

    return recon

def test():

    # data_dir = '/Volumes/T7 Shield/FastMRI/knee/singlecoil_train'
    data_dir = '/Users/alex/Desktop/fastMRI/knee_singlecoil_train'
    # data_dir = '/home/alex/Documents/research/mri/knee_singlecoil_train'
    fnames = glob.glob(data_dir +'/*')

    file_num = 1
    slice_num = 22
    left_idx = 18
    right_idx = 350
    with h5py.File(fnames[file_num], 'r') as hf:
        ks = hf['kspace'][slice_num]
        ks_mask = ks[:, left_idx:right_idx]
    
    mval = np.max(np.abs(ks))
    ks /= mval
    view_im(ks)

    sImg = ks.shape
    m = vdSampleMask(sImg, [50, 40], 0.2 * np.prod(sImg), 'laplace')
    plt.spy(m)
    plt.show()

    w = wtDaubechies2(ks)
    plt.imshow(np.abs(w))
    plt.show()

    undersampled_kspace = ks * m
    view_im(undersampled_kspace)

    return 0

def test_adjoint(x0, f, ip = lambda x, y: np.vdot(x, y), num_test = 10):
    """
    test whether the adjoint of f is implemented correctly
    f should be a function that takes in x0 and an optional parameter of either 'transp' or 'notransp'
    ip is the inner product to use, this is really only for funky situations where you have a real scalar field etc.
    """

    fx0 = f(x0)
    ftfx0 = f(fx0, 'transp')
    rng = np.random.default_rng(20250303)

    error = 0

    dataComplex = False
    if np.any(x0.imag):
        dataComplex = True


    for _ in range(num_test):
        y1 = rng.normal(size=x0.shape)
        y2 = rng.normal(size=fx0.shape)

        if dataComplex:
            y1 = y1 + 1j * rng.normal(size=x0.shape)
            y2 = y2 + 1j * rng.normal(size=fx0.shape)

        fy1 = f(y1)
        fty2 = f(y2, 'transp')
        
        t1 = ip(y1, fty2)
        t2 = ip(fy1, y2)

        error += np.abs(t1 - t2)

    error /= num_test
    assert error < 1e-8, "adjoint test failed"

def test_sense(kSpace):
    """
    SENSE is a model based reconstruction algorithm for undersampled MRI images.
    raa
    TODO: write this lmao
    we'll write this in numpy bc we don't need torch stuff

    minimizes
    || F S x - kSpace ||_2^2
    where S is the coil sensitivities, F is the Fourier transform, and kSpace is the undersampled k-space data
    although here I guess we assume that kSpace is the correct size and just has zeros in the missing entries
    
    oh right the k-space here is going to be [N M Q] where Q is number of coils
    and sense makes are the same size

    """

    rng = np.random.default_rng(20250303)
    # sMaps = rng.random(size = [3, 3, 8]) + 1j * rng.random(size = [3, 3, 8])
    # data = sio.loadmat('/Users/alex/Documents/MATLAB/SFtest.mat')
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
    sMaps = data['sMaps']
    x0 = data['x0']
    sx0 = data['Sx0']

    def applyF(x, op='notransp'):
        if op == 'transp':
            out = x.size * np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( x ) ) )
        else:
            out = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( x ) ) )
        return out

    def applyF_orth(x, op='notransp'):
        if op == 'transp':
            out = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( x ), norm='ortho' ) )
        else:
            out = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( x ), norm='ortho' ) )
        return out
    
    def applyS(x, op='notransp'):
        """
        x should be [N x M] where smaps are [N x M x Q] where Q is number of coils
        """
        if op == 'transp':
            out = np.sum( np.conj(sMaps) * x, 2 )
        else:
            xi = np.expand_dims(x, 2)
            out = sMaps * xi
            out = out.reshape(sMaps.shape)
        return out

    # x = rng.random(size=[50, 1])
    # test_adjoint(x, applyF)

    # x = rng.random(size=[50, 50]) + 1j * rng.random(size=[50, 50])
    # ip = lambda x, y: np.trace(np.dot(np.conj(x.T), y))
    # test_adjoint(x, applyF, ip)

    # x = rng.random(size = [45, 45])
    # test_adjoint(x, applyS)

    # data_mask = np.abs(kSpace) > 0
    # b = kSpace[data_mask]

    def applySF(x, op='notransp'):
        if op == 'transp':
            o1 = applyF(x, 'transp')
            out = applyS(o1, 'transp')
        else: 
            o1 = applyS(x)
            out = applyF(o1)
        return out
        
    y0 = data['y0']
    sty0 = data['Sty0']
    st = applyS(y0, 'transp')
    ip = lambda x, y: np.trace(np.dot(np.conj(x.T), y))
    ip2 = lambda x, y: np.real(np.conj(y.flatten().T) @ x.flatten())
    x = rng.random(size=[3, 3]) + 1j*rng.random(size=[3, 3])
    test_adjoint(x, applySF,ip2)

    return 0

def test_wavelets():
    # rng = np.random.default_rng(20250319)
    # sImg = (256, 256)
    # wavSplit = makeWavSplit(sImg)

    # x = rng.normal(size=sImg) + 1j*rng.normal(size=sImg)
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/wavelet_test.mat')
    a = data['a']
    wavSplit = data['wavSplit']
    wa = wtDaubechies2(a, wavSplit)
    wa2 = data['wa']
    print(np.linalg.norm(wa - wa2))
    iwa = iwtDaubechies2(wa, wavSplit)
    print(np.linalg.norm(iwa - data['iwa']))


def test_funs():
    a = np.ones([15, 15])
    b = upsample2(a, np.array([1, 2]))
    print(b)

def main():
    """
    testing mri utilities
    """
    # test_funs()
    # test_sense(np.random.randn(640, 368, 8))
    test_wavelets()
    
        

if __name__ == "__main__":
    # m = vdSampleMask([640, 368], [50, 40], int(640*368*0.25), 'laplace')
    # plt.spy(m)
    # plt.show()
    main()