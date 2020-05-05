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


def getTrnData(fname,nImg=50,sigma=0.0):
    '''

    Parameters
    ----------
    fname : The demo dataset file
    nImg : number of images. This file has 50 maximum images.
    sigma : The noise standard deviation

    Returns
    -------
    org : pre-calculated MUSSELS reconstruction as ground truth.
    atb : Simulate the 4-shot phase corrupted regridding reconstruction. A^H(b)
          This is the network input.
    csm : Coil sensitivity maps.
    mask : 4-shot data mask.

    '''

    tmp = np.load(fname, mmap_mode='r')
    org=tmp['trnOrg']
    csm=tmp['trnCsm']
    mask=tmp['trnMask']

    atb=np.zeros_like(org)
    for i in range(org.shape[0]):
        tmp=epiA_np(org[i],csm[i],mask[i])
        noise=1j*np.random.randn(*tmp.shape)
        noise+=np.random.randn(*tmp.shape)
        tmp=tmp+noise*sigma
        atb[i]=epiAt_np(tmp,csm[i],mask[i])

    return org,atb,csm,mask

def getTstData(fname,sigma=0.0):
    '''
    Parameters
    ----------
    fname : TYPE
        The is the dataset file
    sigma : TYPE, optional
        standard deviation of the Gaussian noise

    Returns
    -------
    org : load MUSSELS reconstruction for comparison only.
    atb : This is phase corrupted re-gridding output A^H(b)
    csm : coil sensitivity maps
    mask : 4-shot mask
    '''

    tmp = np.load(fname, mmap_mode='r')
    org=tmp['tstOrg']
    csm=tmp['tstCsm']
    mask=tmp['tstMask']
    tmp=epiA_np(org[0],csm[0],mask[0])
    noise=1j*np.random.randn(*tmp.shape)
    noise+=np.random.randn(*tmp.shape)
    tmp=tmp+noise*sigma
    atb=epiAt_np(tmp,csm[0],mask[0])
    atb=atb[None]
    return org,atb,csm,mask


