import numpy as np
import os
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model, model_from_json
import keras

'''

*** this is the architecture that was outlined in the paper ***

layer    type    input size          maps    size    stride     pad
0        input   c x 11 x 11 x 11    -        -       -         -
1        conv    c x 11 x 11 x 11    32       3^3     1^3       1^3
2        mp      32 x 5 x 5 x 5      -        2^3     2^3       0
3        conv    64 x 5 x 5 x 5      64       3^3     1^3       1^3
4        mp      64 x 2 x 2 x 2      -        2^3     2^3       0
5        fc      256                 256      1       -         -
6        soft    2                   2        1       -         -

'''

class cnn_model(object):
    
    def __init__(self, mode='train', patch_size=(11, 11, 11),
                 num_channels=1, name='mdl', path='none'):
        self.name = name
	self.history = 0
        if (mode == 'train'):
            self.patch_size = patch_size
            self.num_channels = num_channels
            self.model = False
            self.build_graph()
            self.compile_graph()
            print('CNN model created')
        elif (mode == 'load'):
            self.path = path
            self.load_model()
            print('model loaded')
        elif (mode == 'classify'):
            self.path = path
            self.load_model()
            self.classify_3d_scan()

    def build_graph(self):

        print('defining model')

        # input layer
        input_layer = Input(np.hstack((self.patch_size, self.num_channels)))
    
        # convolutional layer
        conv_layer1 = Conv3D(filters=32, kernel_size=(3, 3, 3), 
                             strides=(1, 1, 1), padding='valid', 
                             activation='relu')(input_layer)
    
        # max pooling layer
        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2),
                                   strides=(2, 2, 2))(conv_layer1)
    
        # convolutional layer
        conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 3),
                             strides=(1, 1, 1), padding='valid', 
                             activation='relu')(pooling_layer1)
    
        # max pooling layer
        pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2),
                                   strides=(2, 2, 2))(conv_layer2)
    
        # to avoid vanishing gradient
        # perform batch normalization on the convolution outputs before MLP
        pooling_layer2 = BatchNormalization()(pooling_layer2)

        # flatten layer
        flatten_layer = Flatten()(pooling_layer2)

        # dense layer
        dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)

        # dropout (added 4/30/19)
        dropout_layer = Dropout(0.5)(dense_layer1)

        # output layer
        output_layer = Dense(units=2, activation='softmax')(dropout_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)

    def compile_graph(self):

        print('compiling graph')
        self.model.compile(loss=categorical_crossentropy,
                           optimizer=Adadelta(lr=0.1),
                           metrics=['acc'])
    
    def train_network(self, xtrain=0, ytrain=0, batch_size=16,
                      epochs=100, val=0.2):

        print('training model')
        self.model.fit(x=xtrain,
                       y=ytrain,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=val)

        self.history = self.model.history.history
        #print(self.history)

        self.save_model()

    def predict_network(self, xpredict=0, batch_size=16):

        print('predicting patches')
        y_hat = self.model.predict(x=xpredict,
                                   batch_size=batch_size)

        return y_hat

    def classify_3d_scan(self, patient=0):

        # we need two models to be loaded for this
        # for each slide, pass all patches through first network
        # for each patch that the first network predicted as positive, pass through
        # the second networ
        #TODO continue here

    def save_model(self):

        print('saving model')

        # save weights
        self.model.save_weights('{}_weights.h5'.format(self.name))

        # save architecture
        with open('{}_architecture.json'.format(self.name), 'w') as f:
            f.write(self.model.to_json())

    def load_model(self):

        alias = 'lv1out_'+self.name
        filepath = os.path.join(os.path.join('..', self.path), alias)

        # load model from JSON file
        with open(filepath+'_architecture.json', 'r') as f:
            self.model = model_from_json(f.read())

        # load weights into model
        self.model.load_weights(filepath+'_weights.h5' .format(self.name))
