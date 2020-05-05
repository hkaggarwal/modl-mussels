"""
Created on Sept 25, 2019

This is the testing code

author: Hemant Kumar Aggarwal
Email: hemantkumar-aggarwal@uiowa.edu
All rights reserved.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import misc as sf
import readData as rd

cwd=os.getcwd()
tf.reset_default_graph()
np.set_printoptions(precision=3,suppress=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True
#%%

dataset_name='test_data.npz'
#dataset_name='/Shared/lss_haggarwal/All_datasets/epi7sub/diffusion_mri_dataset.npz'

model_name='30Jan_0550pm_7lay_60E'


#%%Read the testing data from dataset.hdf5 file
tstOrg,tstAtb,tstCsm,tstMask=rd.getTstData(dataset_name)
nImg=tstAtb.shape[0]

#%% Load existing model. Then do the reconstruction
z0=np.empty_like(tstAtb)

print ('Now loading the model ...')
modelDir= cwd+'/trained_model/'+model_name #complete path

tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(modelDir)

with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    predT =graph.get_tensor_by_name('predTst:0')
    atbT=graph.get_tensor_by_name('atb:0')
    csmT=graph.get_tensor_by_name('csm:0')
    maskT=graph.get_tensor_by_name('mask:0')
    learned_wts=sess.run(tf.global_variables())

    for i in range(nImg):
        fd={atbT:tstAtb[[i]],csmT:tstCsm[[i]],maskT:tstMask[[i]]}
        z0[i]=sess.run(predT,feed_dict=fd)

print('Reconstruction done')

#%% take some of square for visualization
fun=lambda x:sf.sos(sf.ifft2c(x))
normOrg=fun(tstOrg)
normAtb=fun(tstAtb)
normX1=fun(z0)

#%% Display the output images
print('Now showing output')
plt.figure(figsize=(9,3.5))
plt.subplot(131)
plt.imshow(normOrg[0],'gray')
plt.axis('off')
plt.title('Ground Truth, MUSSLES')
plt.subplot(132)
plt.imshow(normAtb[0],'gray')
plt.title('Network Input')
plt.axis('off')
plt.subplot(133)
plt.imshow(normX1[0],'gray')
plt.axis('off')
plt.title('Network Output, MoDL-MUSSELS')
plt.subplots_adjust(left=0, right=1, top=.93, bottom=0,wspace=0.01)
plt.show()