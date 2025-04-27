import numpy as np
import utils
import matplotlib.pyplot as plt
import opt_algs

def ssqRecon(kSpace):
    """
    MRI recon using the sum of squares method

    INPUTS:
        kSpace: [Nx x Ny x Coils] numpy array of k-space data
    OUTPUTS:
        recon: [Nx x Ny] numpy array of reconstructed image
    """

    coilRecons = ifftRecon(kSpace)
    tmpRecon = coilRecons * np.conj(coilRecons)
    recon = np.sqrt(np.sum(tmpRecon, axis=2))

    return recon

def ifftRecon(kSpace):
    """
    mri recon using the inverse Fourier transform

    INPUTS:
        kSpace: [Nx x Ny x Coils] numpy array of k-space data
    OUTPUTS:
        recon: [Nx x Ny x Coils] numpy array of reconstructed image
    """

    recon = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( kSpace, axes=[0, 1]), axes=[0, 1] ), axes=[0, 1] )

    return recon

def roemerRecon(coilRecons, sMaps = np.array([])):
    """
    MRI recon using the Roemer method

    INPUTS:
        kSpace: [Nx x Ny x Coils] numpy array of k-space data
        sMaps: [Nx x Ny x Coils] numpy array of sensitivity maps
                Typically estimated by PISCO or ESPIRiT
    OUTPUTS:
        recon: [Nx x Ny] numpy array of reconstructed image
    """

    sImg = coilRecons.shape
    nCoils = sImg[2]

    if sMaps.size == 0:
        sqrecon = ssqRecon(coilRecons)
        sMaps = np.zeros_like(coilRecons)
        
        np.divide(coilRecons, sqrecon, out=sMaps, where=sqrecon!=0)

    recon = np.sum( coilRecons * np.conj(sMaps), axis=2 )

    return recon

def compressedSensingRecon(kSpace, wavSplit = []):
    """
    MRI recon using compressed sensing

    solves the problem
    min || W x ||_1 s.t. || MFx - kSpace ||_2 < epsilon

    unconstrained version
    min 0.5 || MFx - kSpace ||_2^2 + lambda || W x ||_1

    where M is the masking operator, F is the Fourier operator, W is the wavelet operator,
    and x is the image in image space

    this is single coil reconstruction
    INPUTS:
        kSpace: [Nx x Ny] numpy array of k-space data
    OUTPUTS:
        recon: [Nx x Ny] numpy array of reconstructed image
    """

    sImg = kSpace.shape

    if len(wavSplit) == 0:
        wavSplit = utils.makeWavSplit(sImg)

    def applyW(x, op='notransp'):
        """
        apply wavelet operator
        """
        if op == 'transp':
            out = utils.iwtDaubechies2(x, wavSplit)
        else:
            out = utils.wtDaubechies2(x, wavSplit)
        return out
    
    mask = np.abs(kSpace) > 0 # find nonzero
    b = kSpace[mask] # b is the measured k-space data

    def applyF_orth(x, op='notransp'):
        if op == 'transp':
            out = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( x ), norm='ortho' ) )
        else:
            out = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( x ), norm='ortho' ) )
        return out

    def applyF(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[0] * x.shape[1]) * np.fft.fftshift( np.fft.ifft2( np.fft.ifftshift( x, axes=[0, 1] ), axes=[0,1] ), axes=[0,1] )
        else:
            out = np.fft.fftshift( np.fft.fft2( np.fft.ifftshift( x, axes=[0,1] ), axes=[0,1] ), axes=[0,1] )
        return out
    
    def applyA(x, op='notransp'):
        if op =='transp':
            out = np.zeros(sImg, dtype=np.complex128)
            out[mask] = x
            out = applyF(out, 'transp')

        else:
            out = applyF(x)
            out = out[mask]
        
        return out
    
    lambd = 1e-8

    x0 = np.random.randn(*kSpace.shape) + 1j*np.random.randn(*kSpace.shape)
    ip = lambda x, y: np.real( np.vdot( x.flatten(), y.flatten() ))
    err1 = utils.test_adjoint(x0, applyW, ip)
    err2 = utils.test_adjoint(x0, applyA, ip)
    err3 = utils.test_adjoint(x0, applyF_orth, ip)
    err4 = utils.test_adjoint(x0, applyF, ip)

    Atb = applyA(b, 'transp')

    # normEst = utils.powerIter(applyA, x0)

    gradf = lambda x: applyA(applyA(x), 'transp') - Atb
    proxg = lambda x, gamma: x + applyW(utils.proxL1(applyW(x), gamma, lambd) - applyW(x), 'transp')

    f = lambda x: 0.5*np.linalg.norm(applyA(x) - b)**2
    g = lambda x: lambd*np.linalg.norm(applyW(x), 1)
    objFun = lambda x: f(x) + g(x)

    x0 = utils.kspace_to_imspace(kSpace)

    xstar, objVals = opt_algs.prox_grad(x0, gradf, proxg, objFun=objFun, maxIter=100, gamma=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('objective values')
    ax.plot(objVals)
    ax.set_xlabel('iteration')
    ax.set_ylabel('objective value')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(np.abs(xstar), cmap='gray')
    plt.show()

def compressedSensingRecon_wc(kSpace, wavSplit = []):
    """
    MRI recon using compressed sensing solving for the wavelet coefficients

    solves the problem
    min || W x ||_1 s.t. || MFx - kSpace ||_2 < epsilon

    unconstrained version
    min 0.5 || MFW^-1x - kSpace ||_2^2 + lambda || z ||_1

    where M is the masking operator, F is the Fourier operator, W is the wavelet operator,
    and x is the wavelet coefficients

    this is single coil reconstruction
    INPUTS:
        kSpace: [Nx x Ny] numpy array of k-space data
    OUTPUTS:
        recon: [Nx x Ny] numpy array of reconstructed image
    """

    sImg = kSpace.shape

    if len(wavSplit) == 0:
        wavSplit = utils.makeWavSplit(sImg)

    def applyW(x, op='notransp'):
        """
        apply wavelet operator
        """
        if op == 'transp':
            out = utils.iwtDaubechies2(x, wavSplit)
        else:
            out = utils.wtDaubechies2(x, wavSplit)
        return out
    
    mask = np.abs(kSpace) > 0 # find nonzero
    b = kSpace[mask] # b is the measured k-space data

    def applyF_orth(x, op='notransp'):
        if op == 'transp':
            out = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( x ), norm='ortho' ) )
        else:
            out = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( x ), norm='ortho' ) )
        return out

    def applyF(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[0] * x.shape[1]) * np.fft.fftshift( np.fft.ifft2( np.fft.ifftshift( x, axes=[0, 1] ), axes=[0,1] ), axes=[0,1] )
        else:
            out = np.fft.fftshift( np.fft.fft2( np.fft.ifftshift( x, axes=[0,1] ), axes=[0,1] ), axes=[0,1] )
        return out
    
    def applyA(x, op='notransp'):
        if op =='transp':
            out = np.zeros(sImg, dtype=np.complex128)
            out[mask] = x
            out = applyF(out, 'transp')
            out = applyW(out)

        else:
            out = applyW(x, 'transp')
            out = applyF(out)
            out = out[mask]
        
        return out
    
    lambd = 0.1

    x0 = np.random.randn(*kSpace.shape) + 1j*np.random.randn(*kSpace.shape)
    ip = lambda x, y: np.real( np.vdot( x.flatten(), y.flatten() ))
    err1 = utils.test_adjoint(x0, applyW, ip)
    err2 = utils.test_adjoint(x0, applyA, ip)
    err3 = utils.test_adjoint(x0, applyF_orth, ip)
    err4 = utils.test_adjoint(x0, applyF, ip)

    Atb = applyA(b, 'transp')

    normEst = utils.powerIter(applyA, x0)

    gradf = lambda x: applyA(applyA(x), 'transp') - Atb
    proxg = lambda x, gamma: utils.proxL1(x, gamma, lambd)

    f = lambda x: 0.5*np.linalg.norm(applyA(x) - b)**2
    g = lambda x: lambd*np.linalg.norm(x, 1)
    objFun = lambda x: f(x) + g(x)

    # x0 = utils.kspace_to_imspace(kSpace)
    # w0 = applyW(x0)
    x0 = np.zeros_like(kSpace, dtype=np.complex128)
    xstar, objVals = opt_algs.prox_grad(x0, gradf, proxg, objFun=objFun, maxIter=100, gamma=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('objective values')
    ax.plot(objVals)
    ax.set_xlabel('iteration')
    ax.set_ylabel('objective value')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(np.abs(applyW(xstar, 'transp')), cmap='gray')
    plt.show()

    

def modelBasedRecon(kSpace, sMaps):
    """
    MRI sense style model based reconstruction

    Solves problem
    min 0.5 * || MFS * x - kSpace ||_2^2
    where M is the masking operator, F is the Fourier operator, S is the sensitivity map operator,
    and x is the image in image space
    INPUTS:
        kSpace: [Nx x Ny x Coils] numpy array of k-space data
        sMaps: [Nx x Ny x Coils] numpy array of sensitivity maps
                Typically estimated by PISCO or ESPIRiT
    OUTPUTS:
        recon: [Nx x Ny] numpy array of reconstructed image


    it really was just a scaling issue
    need to write a gradient descent line search
    """

    sKspace = kSpace.shape
    nCoils = sKspace[2]
    sImg = sKspace[0:2]

    dataMask = np.abs(kSpace) > 0 # find nonzero indices

    maskIndices = np.nonzero(dataMask.reshape(-1, 1))[0]
    
    def applyF_orth(x, op='notransp'):
        if op == 'transp':
            out = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( x ), norm='ortho' ) )
        else:
            out = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( x ), norm='ortho' ) )
        return out
    
    def applyF(x, op='notransp'):
        if op == 'transp':
            out = (x.shape[0] * x.shape[1]) * np.fft.fftshift( np.fft.ifft2( np.fft.ifftshift( x, axes=[0, 1] ), axes=[0,1] ), axes=[0,1] )
        else:
            out = np.fft.fftshift( np.fft.fft2( np.fft.ifftshift( x, axes=[0,1] ), axes=[0,1] ), axes=[0,1] )
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
    
    def applySF(x, op='notransp'):
        if op == 'transp':
            o1 = applyF(x, 'transp')
            out = applyS(o1, 'transp')
        else: 
            o1 = applyS(x)
            out = applyF(o1)
        return out
        
    def applyE(x, op='notransp'):
        if op == 'transp':
            tmp = np.zeros(sKspace, dtype=np.complex128)
            tmp = tmp.reshape(-1, 1)
            tmp[maskIndices] = x
            tmp = tmp.reshape(sKspace)
            #np.putmask(tmp, dataMask, x)

            # tmp = tmp.reshape(sKspace) # should be redundant
            out = applySF(tmp, 'transp')
        else:
            x = x.reshape(sImg)
            out = applySF(x)
            # out = out[dataMask]
            out = out.reshape(-1, 1)[maskIndices]
        
        return out.reshape(-1, 1)
    
    # b = applyE(kSpace, 'transp')
    b = kSpace.reshape(-1, 1)[maskIndices]    
    Etb = applyE(b, 'transp')
    def grad(x):
        # tmp = applyE(applyE(x), 'transp') - Etb
        tmp = applyE(applyE(x), 'transp') - Etb
        return tmp
    
    def obj(x):
        return 0.5 * np.linalg.norm(applyE(x) - b)**2
    
    # solve problem via gradient descent
    x0 = np.zeros(sImg, dtype=np.complex128)
    xi = x0.reshape(-1, 1)

    maxIter = 20

    xtest = np.random.randn(*sImg) + 1j* np.random.randn(*sImg)
    ip = lambda x, y: np.real( np.vdot( x.reshape(-1, 1), y.reshape(-1, 1) ) ) 
    utils.test_adjoint(xtest, applyF, ip)
    utils.test_adjoint(xtest.reshape(-1, 1), applyE, ip)

    nTest = np.random.normal(size = xi.shape)
    print(f'****** POWER ITERATION *******')
    normEst = utils.powerIter(applyE, nTest)
    print(f'norm est of e: {normEst}')
    # alpha = 0.9 / normEst
    linesearch = True
    max_linesearch_iter = 100
    alpha_bar = 1/normEst
    alpha = alpha_bar
    rho = 0.9
    c = 0.9 # linesearch param
    tau = 1.1
    for i in range(maxIter):
        objVal = obj(xi)
        print('Iteration %d: Objective value = %f, alpha: %f' % (i, objVal, alpha))
        if linesearch:
            linesearch_iter = 0
            obj_x = obj(xi)
            ggrad_xi = grad(xi)
            while linesearch_iter < max_linesearch_iter:
                linesearch_iter += 1
                xNew = xi - alpha*ggrad_xi
                obj_xnew = obj(xNew)
                if obj_xnew < obj_x - alpha * c * np.linalg.norm(ggrad_xi.reshape(-1, 1))**2:
                    break
                alpha *= rho
            xi = xNew
            alpha *= tau 
        else:
            alpha = 1e-5
            xi = xi - alpha * grad(xi)

    recon = xi.reshape(sImg)
    plt.imshow(np.abs(recon), cmap='gray')
    plt.show()

def test_singlecoil():
    import scipy.io as sio
    # data = sio.loadmat('/Users/alex/Documents/MATLAB/pdhg_twoLS/brain.mat')
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
    kSpace = data['d2']
    kSpace = kSpace / np.max(np.abs(kSpace))
    sMaps = data['smap']
    sMaps = sMaps / np.max(np.abs(sMaps))

    im2 = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( kSpace, axes=(0,1)), axes=(0,1)), axes=(0,1))
    recon = utils.mri_reconRoemer(im2, sMaps)

    ks_sc = np.fft.fftshift( np.fft.fftn( np.fft.ifftshift( recon, axes=(0,1)), axes=(0,1)), axes=(0,1))

    compressedSensingRecon(ks_sc)
    return 0

def test():
    import scipy.io as sio
    # data = sio.loadmat('/Users/alex/Documents/MATLAB/pdhg_twoLS/brain.mat')
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
    kSpace = data['d2']
    kSpace = kSpace / np.max(np.abs(kSpace))
    sMaps = data['smap']
    sMaps = sMaps / np.max(np.abs(sMaps))

    mask = utils.vdSampleMask(kSpace.shape[0:2], [30, 30], np.round(np.prod(kSpace.shape[0:2]) * 0.05))

    us_kSpace = kSpace*mask[:, :, np.newaxis]
    # compressedSensingRecon(us_kSpace)
    modelBasedRecon(us_kSpace, sMaps=sMaps)
    return 0

def test_mbr():
    import scipy.io as sio
    data = sio.loadmat('/Users/alex/Documents/MATLAB/pdhg_twoLS/mbr_test2.mat')
    data2 = sio.loadmat('/Users/alex/Documents/MATLAB/pdhg_twoLS/mbr_test4.mat')
    kSpace = data2['kData']
    sMaps = data['smap']

    mask = data['wavMask']

    modelBasedRecon(kSpace, sMaps=sMaps)

def test_cs():
    import scipy.io as sio
    # data = sio.loadmat('/Users/alex/Documents/MATLAB/pdhg_twoLS/brain.mat')
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/brain_data.mat')
    kSpace = data['d2']
    kSpace = kSpace / np.max(np.abs(kSpace))
    sMaps = data['smap']
    sMaps = sMaps / np.max(np.abs(sMaps))

    mask = utils.vdSampleMask(kSpace.shape[0:2], [30, 30], np.round(np.prod(kSpace.shape[0:2]) * 0.025))

    im = utils.kspace_to_imspace(kSpace)

    im_roemer = utils.mri_reconRoemer(im, sMaps)
    kSpace2 = utils.imspace_to_kspace(im_roemer)
    kSpace2 = kSpace2 * mask

    utils.view_im(kSpace2)

    compressedSensingRecon_wc(kSpace2)

if __name__ == '__main__':
    # test_singlecoil()
    test_cs()