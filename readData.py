"""
Created on Sept 25, 2019

Read the training and testing data

author: Hemant Kumar Aggarwal
Email: hemantkumar-aggarwal@uiowa.edu
All rights reserved.
"""
import numpy as np
import misc as sf


def epiA_np(ksp,csm,mask):
    img=sf.ifft2c(ksp)
    coilImages=csm* img[:,np.newaxis]
    data=sf.fft2c(coilImages)
    data=mask[:,np.newaxis]*data
    return data

def epiAt_np(ksp,csm,mask):
    ksp=mask[:,np.newaxis]*ksp
    gdata=sf.ifft2c(ksp)
    data=np.conj(csm)*gdata
    img=np.sum(data,-3)
    kspace=sf.fft2c(img)
    return kspace



def commoncode(org,csm,nShots,sigma):
    '''
    Here we simulate the phase erros and generate the N-shot mask. The function
    name commoncode just means this code is common to both traning and testing.

    Parameters
    ----------
    org : This is a MUSSELS reconstruction treated as ground truth.
    csm : coil sensitivity maps estimated using calibration scan
    nShots : number of shots to simulate (like  3-shots,4-shots)
    sigma : add small amount of Gaussian noise since it is simulation.

    Returns
    -------
    shotsKsp : Shots data in k-space
    atb : Simulated A^H(b), that goes as input to the network
    mask : Simulated N-shots mask.
    '''
    nimg,nrow,ncol=org.shape
    scale=np.sqrt(nrow*ncol)
    #generate random phase
    tmp=np.random.randn(nimg,nShots,3,3) + 1j*np.random.randn(nimg,nShots,3,3)
    tmp=np.angle(np.fft.fft2(tmp,(nrow,ncol))/scale )
    ph=np.exp(1j*tmp).astype(np.complex64)

    #% simulate shots
    shotsImg=np.zeros( (nimg,nShots,nrow,ncol),dtype=np.complex64)
    for p in range(nShots):
        shotsImg[:,p]=ph[:,p]*org
    shotsKsp=sf.fft2c(shotsImg)

    #%Generate masks
    mask=np.zeros((nShots,nrow,ncol))
    for i in range(nShots):
        mask[i,:,i:ncol:nShots]=1
    mask=mask.astype(np.complex64)
    mask=np.repeat(mask[None],nimg,0)

    #Generate A^H(b)
    atb=np.zeros_like(shotsKsp)
    for i in range(nimg):
        b=epiA_np(shotsKsp[i],csm[i],mask[i])
        noise=1j*np.random.randn(*b.shape)
        noise+=np.random.randn(*b.shape)
        noise=(noise*sigma).astype(np.complex64)
        b=b+noise
        atb[i]=epiAt_np(b,csm[i],mask[i])

    return shotsKsp,atb,mask

def getTrnDataNshots(fname, nShots,sigma=0.0):

    tmp = np.load(fname, mmap_mode='r')
    org=tmp['trnOrg']
    csm=tmp['trnCsm']

    shotsKsp,atb,mask=commoncode(org,csm,nShots,sigma)

    return shotsKsp,atb,csm,mask

def getTstDataNshots(fname, nShots,sigma=0.0):

    tmp = np.load(fname, mmap_mode='r')
    org=tmp['tstOrg']
    csm=tmp['tstCsm']

    shotsKsp,atb,mask=commoncode(org,csm,nShots,sigma)

    return shotsKsp,atb,csm,mask


