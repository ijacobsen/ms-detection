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

# choose a patient
patient_list = df.index
patient = patient_list[0]

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

'''    
    if first_iteration:
        xtrain_all = xtrain
        y_train_all = y_train
        first_iteration = False
    else:
        xtrain_all = np.concatenate((xtrain_all, xtrain), 0)
        y_train_all = np.concatenate((y_train_all, y_train), 0)
'''

#%% CNN STUFF

# initiate model
model_name = 'test_model'
model = ml.cnn_model(name=model_name, mode='train')

# train model
model.train_network(xtrain=xtrain, ytrain=y_train, batch_size=2, epochs=20)

