'''
this script is used to troubleshoot why every patch is evaluating as 1

1- train a network
2- save the network
3- evaluate the network on the patient that was used to train
4- evaluate the network on a different patient

'''

import keras
import numpy as np
import model_lib as ml
import data_handler as dh
import os

patch_size = (11, 11, 11)
num_channels = 1

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# iterate over patients
patient_list = df.index
patient = patient_list[0]

# get patches
ex = dh.patcher(patch_size=patch_size)
ex.patchify(path_table=df, patient=patient)
patches = ex.patches_xyz

# stack example patches to feed into NN
x_train = [ptch.array for ptch in patches]
xtrain = np.ndarray((len(x_train),
                     x_train[0].shape[0],
                     x_train[0].shape[1],
                     x_train[0].shape[2],
                     num_channels))
ytrain = [int(ptch.label) for ptch in patches]

# TODO this is very sloppy... xtrain[i, :, :, :, 0] ... 0 is hardcoded
# in to account for only 1 channel (modality) being used... fix this!
# fill xdata with patch data
for i in range(len(xtrain)):
    xtrain[i, :, :, :, 0] = x_train[i]
ytrain = np.array(ytrain)

# convert target variable into one-hot
y_train = keras.utils.to_categorical(ytrain, 2)

# its easy
xtrain_all = xtrain
ytrain_all = y_train

# initiate model
model_name = 'trouble_{}'.format(patient)
model = ml.cnn_model(name=model_name, mode='train')

# train model
model.train_network(xtrain=xtrain_all, ytrain=ytrain_all, 
                    batch_size=8, epochs=100)
    
# load model
# model = ml.cnn_model(name=model_name, mode='load')

