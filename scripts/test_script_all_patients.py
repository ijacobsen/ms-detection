from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
import numpy as np
import keras

import data_handler as dh
import os

patch_size = (11, 11, 11)
num_channels = 1

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [ di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair') 
print('data loaded')

# choose a patient
patient_list = df.index
patient = patient_list[0]

first_iteration = True
for patient in patient_list:
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
        y_train_all = y_train
        first_iteration = False
    else:
        xtrain_all = np.concatenate((xtrain_all, xtrain), 0)
        y_train_all = np.concatenate((y_train_all, y_train), 0)

#%% CNN STUFF


'''
layer    type    input size          maps    size    stride     pad
0        input   c x 11 x 11 x 11    -        -       -         -
1        conv    c x 11 x 11 x 11    32       3^3     1^3       1^3
2        mp      32 x 5 x 5 x 5      -        2^3     2^3       0
3        conv    64 x 5 x 5 x 5      64       3^3     1^3       1^3
4        mp      64 x 2 x 2 x 2      -        2^3     2^3       0
5        fc      256                 256      1       -         -
6        soft    2                   2        1       -         -
'''

# input layer
input_layer = Input(np.hstack((patch_size, num_channels)))

# convolutional layer
conv_layer1 = Conv3D(filters=32, kernel_size=(3, 3, 3), 
                     strides=(1, 1, 1), padding='valid', 
                     activation='relu')(input_layer)

# max pooling layer
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_layer1)

# convolutional layer
conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 3),
                     strides=(1, 1, 1), padding='valid', 
                     activation='relu')(pooling_layer1)

# max pooling layer
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_layer2)


# ** UNSURE ABOUT THIS ... copying it from example
# perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)


dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
#dense_layer1 = Dropout(0.4)(dense_layer1)

output_layer = Dense(units=2, activation='softmax')(dense_layer1)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])

model.fit(x=xtrain_all, y=y_train_all, batch_size=16, epochs=100, validation_split=0.2)
