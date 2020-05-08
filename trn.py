"""
Created on Sept 25, 2019

This file trn.py is the main training code for MoDL-MUSSELS.
By default, it should run without any changes.

*********** Citation ******************
This code is release to promote the reproducible research.
It complements the following paper:

Title: MoDL-MUSSELS: Model-Based Deep Learning for
       Multishot Sensitivity-Encoded Diffusion MRI
Authors: Hemant K. Aggarwal, Merry Mani, and Mathews Jacob
Journal: IEEE Transactions on Medical Imaging, 39(4), Apr, 2020
DOI: 10.1109/TMI.2019.2946501
arXiv PDF: https://arxiv.org/abs/1812.08115

If you find this code helpful in someway then please consider citing this paper.

************Code Details****************************

At the end of training, it will save the output model in the folder "trained_models".
The trained_model folder will have model saved with current date and time in its name.
Copy that folder_name and paste in the tst.py file, to see the output


author: Hemant Kumar Aggarwal
Email: hemantkumar-aggarwal@uiowa.edu
All rights reserved.
*******************************************************
"""

import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from datetime import datetime
from tqdm import tqdm
from model import modl_mussles #this is the main function
import readData as rd # read the demo dataset


tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

#--------------------Set some parameters------------------
nShots=4 # simulate N-Shots data
epochs=50 #number of training epoches
K=1 #number of unrolls, maximum value of 5 for a 12-GB gpu but 1 should be OK too.
lamI,lamK=.05,.01 # regularization parameters for image and k-space
sigma=.005 #small amount of noise in k-space

# Please download the training dataset (320 MB around) from this link:
#dataset link: https://drive.google.com/open?id=10Blm-wX8ofyqLQ6w1qFcm7P-j5vcus6-

dataset_name='diffusion_mri_dataset.npz' #training dataset full file name
dataset_name='/Shared/lss_haggarwal/All_datasets/epi7sub/diffusion_mri_dataset.npz'

#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='trained_model/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%S%P_")+ \
        str(epochs)+'E_'+str(K)+'K_'+str(nShots)+'Shots'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'

#%% prepare data
trnOrg,trnAtb,trnCsm,trnMask=rd.getTrnDataNshots(dataset_name,nShots,sigma)
nImg=trnOrg.shape[0]
#%% save the test model
#The postfix T stands for a tensor just for nomenlcature
#We had compressed the coil sensitivities to have just 4 coils. Hence fixed.
tf.reset_default_graph()

atbT  = tf.placeholder(tf.complex64,shape=(None,nShots,256,256),name='atb')
csmT  = tf.placeholder(tf.complex64,shape=(None,4,256,256),name='csm')
maskT = tf.placeholder(tf.complex64,shape=(None,nShots,256,256),name='mask')

with tf.device('/gpu:0'):
    predTst=modl_mussles(atbT,csmT,maskT,lamK,lamI,K)
predTst=tf.identity(predTst,name='predTst')
sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved : ' +savedFile)

#%% creating the tensorflow dataset
batchSize=1 #Only 1 is good because N-shots are there in one batch
nTrn=trnOrg.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs


tf.reset_default_graph()
orgP = tf.placeholder(tf.complex64,shape=(None,nShots,256,256),name='org')
atbP  = tf.placeholder(tf.complex64,shape=(None,nShots,256,256),name='atb')
csmP  = tf.placeholder(tf.complex64,shape=(None,4,256,256),name='csm')
maskP = tf.placeholder(tf.complex64,shape=(None,nShots,256,256),name='mask')

trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,csmP,maskP))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=100)
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
iterator=trnData.make_initializable_iterator()
orgT,atbT,csmT,maskT= iterator.get_next('getNext')

#%% make training model
predT=modl_mussles(atbT,csmT,maskT,lamK,lamI,K)
loss= tf.reduce_mean(tf.pow(tf.abs(predT-orgT),2))

tf.summary.scalar('loss', loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
opToRun=tf.train.AdamOptimizer().minimize(loss)

#%% training code

print ('*************************************************')
print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))

saver = tf.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    feedDict={orgP:trnOrg,atbP:trnAtb,csmP:trnCsm,maskP:trnMask}
    sess.run(iterator.initializer,feed_dict=feedDict)

    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,update_ops,opToRun])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                totalLoss=[]
        except tf.errors.OutOfRangeError:
            break
    writer.close()
    _=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')
