# MoDL-MUSSELS
Diffusion MRI using Deep Learning

### Reference paper: 

MoDL-MUSSELS: Model-Based Deep Learning for Multi-Shot Sensitivity Encoded Diffusion MRI by H.K. Aggarwal, Marry Mani, Mathews Jacob in IEEE Transactions on Medical Imaging, 39(4), Apr 2020.

PDF Link: https://arxiv.org/abs/1812.08115

IEEE Xplore: https://ieeexplore.ieee.org/document/8863423

#### What this code do:
This code can reconstruct multi-shot diffuion MR images. This code will reduce the phase-artifacts from the multi-shot diffusion data.
In the above paper, we propose a technique to combine the power of deep-learning with the model-based approaches. 
This code utilizes deep learning to accelerate MUSSELS algorithm ( https://doi.org/10.1002/mrm.28090) which is based on structured low rank technique.

#### Output of running the test code:
![alt text](https://github.com/hkaggarwal/modl-mussels/blob/master/output.jpeg)



#### Dependencies

We have tested the code in Anaconda python 3.7 with Tensorflow-1.15.

The training code requires `tqdm` library. It is a nice library that is helpful in tracking the training progress.
It can be installed using:
`conda install tqdm`

In addition, matplotlib is required to visualize the output images.

#### Dataset
This git repository includes test data in the file `test_data.npz`. The testing script `tst.py` will use this image by default and does not require full data download for the testing purpose.
We have release a subset of the dataset, used in the paper, for the training code demo. You can download this data from the below link.

 **Download Link** :  https://drive.google.com/open?id=10Blm-wX8ofyqLQ6w1qFcm7P-j5vcus6-

You will need this file `diffusion_mri_dataset.npz` (320 MB) to run the training code `trn.py`. You can download the dataset from the link provided above.  You do not need to download the `diffusion_mri_dataset.npz` for testing purpose.

You will find the dataset acquisition details in the PDF of the paper (link given above).

`trnOrg`: This is complex arrary of 50x4x256x256 containing 50 directions from different subjects and slices for the training            purpose. Each direction has 4-shots of size 256x256. These are the MUSSELS reconstructions treated as ground truth.
        
`trnCsm`: This is a complex array of 50x4x256x256 representing coil sensitity maps (csm). Here 4 represent number of coils after coil compression. Please note that original rawdata had 32 coils. The code MoDL-MUSSELS can be trained with 4 coils and later tested with 32 coils as well.

`trnMask`: This is a 4-shot undersampling mask.

`tstOrg`,`tstCSM`, `tstMask`: These are similar arrays for testing purpose.


#### How to run the code

First, ensure that Tensorflow 1.15 is installed and working with GPU. The code may work with other versions of the TensorFlow as well. But we did our testings with version 1.15.
Second, just clone or download this reporsitory. The `tst.py` file should run without any changes in the code.
On the command prompt `cd` to this cloned `modl-mussels` directory i.e. the directory containig `tst.py`.
Then you can run the test code using the command: 

`$python tst.py` from the command prompt. This will load the pre-trained model from the directory `trained-model`.


#### Files description
The folder `trained-model` contain the learned tensorflow model parameters. `tst.py` will use it to read the model and run on the demo image in the file `test_data.npz`. 

`misc.py`: This file contain some supporting functions.

`model.py`: This file contain the code for creating the MoDL-MUSSELS architecture. Please note that it has a  conjugate-gradient code that will run simultaneously on all 4-shots on GPU on complex data.
	      
`trn.py`: This is the training code.

`tst.py`: This is the testing code.


#### Contact
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at jnu.hemant@gmail.com.
