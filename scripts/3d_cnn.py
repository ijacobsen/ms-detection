from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
import plotly.graph_objs as go
from matplotlib.pyplot import cm
import numpy as np
import keras
import h5py

import data_handler as dh
import os

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

# get patches
ex = dh.patcher(patch_size = (9, 25, 25))
ex.patchify(path_table=df, patient=patient)
patches = ex.patches





with h5py.File('../ex_data/full_dataset_vectors.h5', 'r') as dataset:
    x_train = dataset["X_train"][:] # (10000, 4096)
    x_test = dataset["X_test"][:]   # (2000, 4096)
    y_train = dataset["y_train"][:] # (10000, 10)
    y_test = dataset['y_test'][:]   # (2000, 10)

xtrain = np.ndarray((x_train.shape[0], 4096, 3)) # (10000, 4096, 3)
xtest = np.ndarray((x_test.shape[0], 4096, 3))   # (2000, 4096, 3)

''' idk what this function and for loops do '''
## iterate in train and test, add the rgb dimention 
def add_rgb_dimention(array):
    scaler_map = cm.ScalarMappable(cmap="Oranges")
    array = scaler_map.to_rgba(array)[:, : -1]
    return array
for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimention(x_train[i])
for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimention(x_test[i])
    
## convert to 1 + 4D space (first argument represents number of rows in the dataset)
xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3) # (10000, 16, 16, 16, 3)
xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)    # (2000, 16, 16, 16, 3)

# 10000 training examples, image size 16x16x16, 3 channels


# batch patch arrays
xtrain = [np.hstack(xtrain, )]

## convert target variable into one-hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

## input layer
input_layer = Input((16, 16, 16, 3))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=10, activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
model.fit(x=xtrain, y=y_train, batch_size=128, epochs=50, validation_split=0.2)
