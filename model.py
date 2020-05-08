"""
Created on Sept 25, 2019

Create the model architecture

author: Hemant Kumar Aggarwal
Email: hemantkumar-aggarwal@uiowa.edu
All rights reserved.
"""

import tensorflow as tf
import misc as sf
from os.path import expanduser
home = expanduser("~")


#%%

def convLayer(x, szW,training,i):
    with tf.name_scope('layers'):
        with tf.variable_scope('lay'+str(i)):
            W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
            y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
            if training!='linear':
                #y=tf.nn.relu(y)
                y=tf.nn.leaky_relu(y,alpha=.5)
    return y


def smallModel(inp,c,training):
    fs=3 #filter size
    with tf.name_scope('dwModl'):
        x=convLayer(inp,(fs,fs,c ,64),training,1)
        x=convLayer(x,(fs,fs,64,64),training,2)
        x=convLayer(x,(fs,fs,64,128),training,3)
        x=convLayer(x,(fs,fs,128,128),training,4)
        x=convLayer(x,(fs,fs,128,64),training,5)
        x=convLayer(x,(fs,fs,64,64),training,6)
        x=convLayer(x,(1,1,64,c),'linear',7)

    return x
#%%
def epiA(ksp,csm,mask):
    with tf.name_scope('epiA'):
        img=sf.tf_ifft2c(ksp)
        coilImages=csm* img[:,tf.newaxis]
        data=sf.tf_fft2c(coilImages)
        data=mask[:,tf.newaxis]*data
        return data

def epiAt(ksp,csm,mask):
    with tf.name_scope('epiAt'):
        ksp=mask[:,tf.newaxis]*ksp
        gdata=sf.tf_ifft2c(ksp)
        data=tf.conj(csm)*gdata
        img=tf.reduce_sum(data,-3)
        kspace=sf.tf_fft2c(img)
    return kspace

def cg4shots(B,rhs,maxIter,cgTol,x):
    #This CG works on all N-shots simultaneously for speed
    with tf.name_scope('myCG'):
        one=tf.constant(1)
        zero=tf.constant(0)
        cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,maxIter), tf.sqrt(tf.reduce_min(tf.abs(rTr)))>cgTol)
        fn=lambda x,y: tf.reduce_sum(tf.conj(x)*y,axis=(-1,-2),keepdims=True)
        def body(i,rTr,x,r,p):
            with tf.name_scope('cgBody'):
                Ap=B(p)
                alpha = rTr / fn(p,Ap)
                x = x + alpha * p
                r = r - alpha * Ap
                rTrNew = fn(r,r)
                beta = rTrNew / rTr
                p = r + beta * p
            return i+one,rTrNew,x,r,p
        i=zero
        r=rhs-B(x)
        p=r
        rTr = fn(r,r)
        loopVar=i,rTr,x,r,p
        out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return out

#%%

def Dw(inp):

    with tf.name_scope('myModel'):
        inp1=sf.c2rT(inp)
        c=inp1.shape.as_list()[-1]
        mn=tf.reduce_mean(inp1,axis=(-2,-3))
        st=tf.keras.backend.std(inp1)
        tfn=tf.newaxis
        inp1=(inp1-mn[:,tfn,tfn])/st
        with tf.variable_scope('unet',reuse=tf.AUTO_REUSE):
            nw=smallModel(inp1,c,True)
            nw=nw+inp1
        mn2=tf.reduce_mean(nw,axis=(-2,-3))
        nw=(nw-mn2[:,tfn,tfn])
        nw=nw*st+mn[:,tfn,tfn]
        nw=sf.r2cT(nw)

    return nw

#%%
def Dc(rhsT,csmT,maskT,xprev,lamKT,lamIT,cgTol,cgIter):

    def fn(tmp):
        rhs,csm,mask,xin=tmp
        A= lambda x: epiA(x,csm,mask)
        At=lambda x: epiAt(x,csm,mask)
        B= lambda x: At(A(x)) + lamKT*x+ lamIT*x
        x=cg4shots(B,rhs,cgIter,cgTol,xin)
        return x

    inp=(rhsT,csmT,maskT,xprev)
    rec=tf.map_fn(fn,inp,dtype=tf.complex64,name='mapFn' )

    return rec

#%%

def modl_mussles(atbT,csmT,maskT,lamK,lamI,K):

    '''
    Parameters
    ----------
    atbT : re-gridding reconstruction tensor
    csmT : coil sensitivity map tensor
    maskT : 4-shot mask
    lamK : k-space regularizer
    lamI : image space regularizer
    K : number of unrolls.

    Returns
    -------
    reconstructed output
    '''

    with tf.name_scope('model'):
        tol=tf.constant(1e-3,dtype=tf.float32)
        cgIter=tf.constant(7)
        lamKT= tf.constant(lamK+0j, dtype=tf.complex64)
        lamIT= tf.constant(lamI+0j, dtype=tf.complex64)
        zero=tf.constant(0.+0j,dtype=tf.complex64)

        xinit=tf.zeros_like(atbT)

        x=Dc(atbT,csmT,maskT,xinit,lamKT,zero,tol,cgIter)

        i=0
        cond=lambda i,*_: tf.less(i,K)
        loopVar=i,x
        def body(i,xin):
            with tf.variable_scope('unetKsp'):
                z1=Dw(xin) #Eq. 25 in the paper
            with tf.variable_scope('unetImg'):
                z2=sf.tf_fft2c(Dw(sf.tf_ifft2c(xin))) #Eq. 25 in the paper

            rhsT=atbT+lamKT*z1 + lamIT*z2
            rec=Dc(rhsT,csmT,maskT,xin,lamKT,lamIT,tol,cgIter) #Eq. 23
            return i+1,rec
        recT=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[1]

    return recT
