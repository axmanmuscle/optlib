"""
mri testing utils
"""
import numpy as np
import scipy.io as sio
import utils

def test_sense():
    data = sio.loadmat('/Users/alex/Documents/School/Research/Dwork/dataConsistency/MB_data.mat')
    kSpace = data['kData']
    sMaps = data['sMaps']
    x0 = data['x0']
    sx0 = data['Sx0']
    sKspace = kSpace.shape
    sImg = sKspace[0:2]
    dataMask = np.abs(kSpace) > 0 # find nonzero indices
    
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
            linearIndices = np.nonzero(dataMask.reshape(-1, 1))[0]
            tmp = tmp.reshape(-1, 1)
            tmp[linearIndices] = x
            tmp = tmp.reshape(sKspace)
            #np.putmask(tmp, dataMask, x)

            # tmp = tmp.reshape(sKspace) # should be redundant
            out = applySF(tmp, 'transp')
        else:
            x = x.reshape(sImg)
            out = applySF(x)
            # out = out[dataMask]
            linearIndices = np.nonzero(dataMask.reshape(-1, 1))[0]
            out = out.reshape(-1, 1)[linearIndices]
        
        return out.reshape(-1, 1)
    
    def applyEF(x, op='notransp'):
        if op == 'transp':
            tmp = np.zeros(sKspace, dtype=np.complex128)
            linearIndicesF = np.nonzero(dataMask.reshape(-1, 1, order='F'))[0]
            tmp = tmp.reshape(-1, 1, order='F')
            tmp[linearIndicesF] = x
            tmp = tmp.reshape(sKspace, order='F')

            # tmp = tmp.reshape(sKspace) # should be redundant
            out = applySF(tmp, 'transp')
        else:
            x = x.reshape(sImg, order='F')
            out = applySF(x)
            # out = out[dataMask]
            linearIndicesF = np.nonzero(dataMask.reshape(-1, 1, order='F'))[0]
            out = out.reshape(-1, 1, order='F')[linearIndicesF]
        
        return out.reshape(-1, 1, order='F')
    
    ip = lambda x, y: np.real( np.vdot( x.flatten(), y.flatten() ) )
    utils.test_adjoint(x0, applyF, ip)
#    utils.test_adjoint(x0, applySF, ip)
    utils.test_adjoint(x0, applyE, ip)
    test1 = np.linalg.norm(applyS(x0) - data['Sx0'])
    test2 = np.linalg.norm(applyF(x0) - data['Fx0'])
    test3 = np.linalg.norm(applySF(x0) - data['SFx0'])
    exo = applyE(x0)
    test4 = np.linalg.norm(exo - data['Ex0'])

    efx0 = applyEF(x0)
    test4b = np.linalg.norm(efx0 - data['Ex0'])

    y0 = data['y0']
    linearIndices = np.nonzero(dataMask.reshape(-1, 1))[0]
    test5 = np.linalg.norm(applyS(y0, 'transp') - data['Sty0'])
    test6 = np.linalg.norm(applyF(y0, 'transp') - data['Fty0'])
    test7 = np.linalg.norm(applySF(y0, 'transp') - data['SFty0'])

    y02 = y0.reshape(-1, 1)[linearIndices]
    ey02 = applyE(y02, 'transp')

    ytest = data['ytest']
    test8 = np.linalg.norm(ey02 - data['Ety0'])

    
    # b = applyE(kSpace, 'transp')
    
    b = kSpace.reshape(-1, 1)[linearIndices]

    linearIndicesF = np.nonzero(dataMask.reshape(-1, 1, order='F'))[0]
    bF = kSpace.reshape(-1, 1, order='F')[linearIndicesF]
    # b = kSpace[dataMask]
    # b = b.flatten()

    Etb = applyE(b, 'transp')
    def grad(x):
        # tmp = applyE(applyE(x), 'transp') - Etb
        tmp = applyE(applyE(x) - b, 'transp')
        return tmp
    
    xr = data['xr']
    exr = data['exr']
    etexr = data['etexr']
    gxr = grad(xr)
    gxr = gxr.reshape(-1, 1)
    test8 = np.linalg.norm(gxr - data['gxr'])

def main():
    test_sense()
    return 0

if __name__ == "__main__":
    main()