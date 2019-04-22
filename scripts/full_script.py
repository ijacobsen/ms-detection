'''
this script will train 15 CNNs, using leave one out
'''

import keras
import numpy as np
import model_lib as ml
import data_handler as dh
import os

# %%    CONFIGURATION
patch_size = (11, 11, 11)
num_channels = 1

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# choose a patient
patient_list = df.index

# %%    LAYER 1 TRAINING
for k in range(len(patient_list)):

    pats_lv1out = [patient_list[j] for j in range(len(patient_list)) if j != k]

    first_iteration = True
    for patient in pats_lv1out:
    
        # get patches
        ex = dh.patcher(patch_size=patch_size)
        ex.patchify(path_table=df, patient=patient)
        patches = ex.patches_xyz
    
        # TODO SHUFFLE PATCHES
    
        # stack example patches to feed into NN
        x_train = [ptch.array for ptch in patches]
        xtrain = np.ndarray((len(x_train),
                             x_train[0].shape[0],
                             x_train[0].shape[1],
                             x_train[0].shape[2],
                             num_channels))
        ytrain = [int(ptch.label) for ptch in patches]
    
        # TODO this is very sloppy... xtrain[i, :, :, :, 0] ... 0 is hardcoded in
        # to account for only 1 channel (modality) being used... fix this!
        # fill xdata with patch data
        for i in range(len(xtrain)):
            xtrain[i, :, :, :, 0] = x_train[i]
        ytrain = np.array(ytrain)
    
        # convert target variable into one-hot
        y_train = keras.utils.to_categorical(ytrain, 2)
    
        # before stacking, 
        # xtrain (500, 11, 11, 11, 1)
        # y_train (500, 2)
    
        if first_iteration:
            xtrain_all = xtrain
            ytrain_all = y_train
            first_iteration = False
        else:
            xtrain_all = np.concatenate((xtrain_all, xtrain), 0)
            ytrain_all = np.concatenate((ytrain_all, y_train), 0)

    ### CNN STUFF
    
    # initiate model
    model_name = 'lv1out_{}'.format(patient_list[k])
    model = ml.cnn_model(name=model_name, mode='train')
    
    # train model
    model.train_network(xtrain=xtrain_all, ytrain=ytrain_all, 
                        batch_size=16, epochs=100)
    
    # load model
    # model = ml.cnn_model(name=model_name, mode='load')

