"""
Created on Aug 6th, 2018

This file contains some supporting functions

@author:Hemant
"""

import numpy as np
import tensorflow as tf


#%%

def fft2c(img):

    shp=img.shape
    nimg=int(np.prod(shp[0:-2]))
    scale=1/np.sqrt(np.prod(shp[-2:]))
    img=np.reshape(img,(nimg,shp[-2],shp[-1]))

    tmp=np.empty_like(img,dtype=np.complex64)
    for i in range(nimg):
        tmp[i]=scale*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img[i])))

    kspace=np.reshape(tmp,shp)
    return kspace


def ifft2c(kspace):

    shp=kspace.shape
    scale=np.sqrt(np.prod(shp[-2:]))
    nimg=int(np.prod(shp[0:-2]))

    kspace=np.reshape(kspace,(nimg,shp[-2],shp[-1]))

    tmp=np.empty_like(kspace)
    for i in range(nimg):
        tmp[i]=scale*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[i])))

    img=np.reshape(tmp,shp)
    return img


def tf_fft2c(kspace):
    shp=tf.shape(kspace)
    scale=tf.sqrt(tf.to_float(shp[-2]*shp[-1]))
    scale=tf.to_complex64(scale)
    shifted=tf_shift2d(kspace)
    xhat=tf.spectral.fft2d(shifted)/scale
    centered=tf_shift2d(xhat)
    return centered


def tf_ifft2c(kspace):
    shp=tf.shape(kspace)
    scale=tf.sqrt(tf.to_float(shp[-2]*shp[-1]))
    scale=tf.to_complex64(scale)
    shifted=tf_shift2d(kspace)
    xhat=tf.spectral.ifft2d(shifted)*scale
    centered=tf_shift2d(xhat)
    return centered
#%% fftshifts on last two dimensions

def getIdx(x):
    xx=np.ceil(x/2).astype(np.int32)
    idx=np.concatenate( (range(xx,x),range(xx)),axis=0)
    return idx

def shift2d(img):
    x,y=img.shape[-2:]
    xid=getIdx(x)
    yid=getIdx(y)
    img=img[...,xid,:]
    img=img[...,yid]
    return img

def tf_getIdx(x):
    two=tf.constant(2)
    xx=tf.cast(tf.ceil(x/two),tf.int32)
    idx=tf.concat([tf.range(xx,x),tf.range(xx)],axis=0)
    return idx

def tf_shift2d(imgT):

    shp=tf.shape(imgT)
    x,y=shp[-2],shp[-1]
    xid=tf_getIdx(x)
    yid=tf_getIdx(y)
    imgT=tf.gather(imgT,xid,axis=-2)
    imgT=tf.gather(imgT,yid,axis=-1)

    return imgT

def tf_fftshift(x):
    shp=x.get_shape().as_list()[-2:]
    dim= [s//2 for s in shp]
    y=tf.manip.roll(x,dim,(-2,-1))
    return y


def tf_ifftshift(x):
    shp=x.get_shape().as_list()[-2:]
    dim= [(s+1)//2 for s in shp]
    y=tf.manip.roll(x,dim,(-2,-1))
    return y


def myfftshift(x):
    shp=x.shape[-2:]
    dim= [s//2 for s in shp]
    y=np.roll(x,dim,(-2,-1))
    return y


def myifftshift(x):
    shp=x.shape[-2:]
    dim= [(s+1)//2 for s in shp]
    y=np.roll(x,dim,(-2,-1))
    return y

#%%
def sos(data,dim=-3):

    res= np.sqrt(np.sum(np.abs(data)**2,dim))
    return res

def tf_sos(data,dim=-3):

    res= tf.sqrt(tf.reduce_sum(tf.abs(data)**2,dim))
    return res
#%% these are some some real to complex (r2c) and complex to real (c2r) functions
def r2c(inp):
    idx=inp.shape[-1]//2
    out=inp[...,0:idx] +1j* inp[...,idx:]
    out=np.transpose(out,(0,3,1,2))
    return out

def c2r(inp):
    inp=np.transpose(inp,(0,2,3,1))
    out=np.concatenate([np.real(inp),np.imag(inp)],axis=-1)
    return out

def r2cT(inp):
    idx=inp.get_shape().as_list()[-1]//2

    out=tf.complex(inp[...,0:idx],inp[...,idx:])
    out=tf.transpose(out,(0,3,1,2))
    return out

def c2rT(inp):
    inp=tf.transpose(inp,(0,2,3,1))
    out=tf.concat([tf.real(inp),tf.imag(inp)],axis=-1)
    return out


#%%

def tf_r2c(inp):
    out=tf.complex(inp[...,0],inp[...,1])
    return out

def tf_c2r(inp):
    out=tf.stack([tf.real(inp),tf.imag(inp)],axis=-1)
    return out

#%%
