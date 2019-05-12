import numpy as np
import os
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model, model_from_json
import keras
import data_handler as dh

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
                 num_channels=1, name='mdl', path='none', lr='0.1'):
        self.name = name
        self.history = 0

        if (mode == 'train'):
            self.patch_size = patch_size
            self.num_channels = num_channels
            self.model = False
            self.lr = lr
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
                           optimizer=Adadelta(lr=self.lr),
                           metrics=['acc'])
    
    def train_network(self, xtrain=0, ytrain=0, batch_size=16,
                      epochs=100, val=0.2):

        print('training model')
        self.btchsz = batch_size
        self.model.fit(x=xtrain,
                       y=ytrain,
                       batch_size=self.btchsz,
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

    def save_model(self):

        print('saving model')

        # save weights
        self.model.save_weights('_weights_lr={}_btch={}.h5'.format(self.name, self.lr, self.btchsz))

        # save architecture
        with open('_architecture_lr={}_btch={}.json'.format(self.name, self.lr, self.btchsz), 'w') as f:
            f.write(self.model.to_json())

    def load_model(self):

        alias = 'lv1out_'+self.name
        filepath = os.path.join(os.path.join('..', self.path), alias)

        # load model from JSON file
        with open(filepath+'_architecture.json', 'r') as f:
            self.model = model_from_json(f.read())

        # load weights into model
        self.model.load_weights(filepath+'_weights.h5' .format(self.name))

#%% TODO work here
class classifier(object):
    
    def __init__(self, mode='classify', patch_size=(11, 11, 11),
                 num_channels=1, name='none', path='none', data='none'):

        self.name = name # patient that was left out
        self.patch_size = patch_size

        if (mode == 'classify'):
            self.df = data
            self.path = path
            self.load_models()
            print('classifier ready')

    def load_models(self):

        print('loading network one')
        alias = 'lv1out_network1_'+self.name
        filepath = os.path.join(os.path.join('..', self.path), alias)

        # load model from JSON file
        with open(filepath+'_architecture.json', 'r') as f:
            self.network1 = model_from_json(f.read())

        # load weights into model
        self.network1.load_weights(filepath+'_weights.h5' .format(self.name))

        print('loading network two')
        alias = 'lv1out_network2_'+self.name
        filepath = os.path.join(os.path.join('..', self.path), alias)

        # load model from JSON file
        with open(filepath+'_architecture.json', 'r') as f:
            self.network2 = model_from_json(f.read())

        # load weights into model
        self.network2.load_weights(filepath+'_weights.h5' .format(self.name))

        print('networks loaded')

    def classify_network1(self, batch_size=128):
        
        print('classifying patches through network 1')
    
        # reformat data to feed into keras
        xdata = self.patches_to_x_array(patches=self.patches)
        
        y_hat = self.network1.predict(x=xdata,
                                      batch_size=batch_size)
        '''
        for debugging without keras
        y_hat = np.random.randint(0, 2, xdata.shape[0])
        y_hat = keras.utils.to_categorical(y_hat, 2)
        '''

        # threshold predictions... [0, 1] is a positive example
        for i in np.arange(len(self.patches)):
            if y_hat[i, 1] > 0.5:
            
                self.patches[i].label = '1'
            else:
                self.patches[i].label = '0'
        
        # store network 1 segmentation [x, y, z, label]... only sotre positive
        n1_slice_seg = [ptch.coords + [ptch.label] for ptch in self.patches if ptch.label == '1']
        self.n1_seg.append(n1_slice_seg)
        
        return 0
    
    def classify_network2(self, batch_size=128):
        
        print('classifying patches through network 2')
        
        # find positive patches from network1
        patches = [ptch for ptch in self.patches if ptch.label == '1']
        
        #coords = [ptch[:3] for ptch in self.n1_seg if ptch[3] == '1']       

        if (len(patches) > 0): 

            # reformat data to feed into keras
            xdata = self.patches_to_x_array(patches=patches)
        
            y_hat = self.network2.predict(x=xdata,
                                          batch_size=batch_size)
            '''
            for debugging without using keras
            y_hat = np.random.randint(0, 2, xdata.shape[0])
            y_hat = keras.utils.to_categorical(y_hat, 2)
            '''

            # threshold predictions... [0, 1] is a positive example
            for i in np.arange(len(patches)):
                if y_hat[i, 1] > 0.5:
                    patches[i].label = '1'
                else:
                    patches[i].label = '0'
        
            # store network 1 segmentation [x, y, z, label]... only save positive patches
            n2_slice_seg = [ptch.coords + [ptch.label] for ptch in patches if ptch.label == '1']
            self.n2_seg.append(n2_slice_seg)
        
        return 0
    
    def patches_to_x_array(self, patches='none'):

            # stack example patches to feed into NN
            x_data = [ptch.array for ptch in patches]
            xdata = np.ndarray((len(x_data),
                                x_data[0].shape[0],
                                x_data[0].shape[1],
                                x_data[0].shape[2],
                                1)) # only one channel
            # fill xdata with patch data
            for i in range(len(xdata)):
                xdata[i, :, :, :, 0] = x_data[i]

            return xdata

    def classify_scan(self, patient=0):

        # we need two models to be loaded for this
        # for each slide, pass all patches through first network
        # for each patch that the first network predicted as positive, pass through
        # the second network

        # prepare data
        self.n1_seg = []
        self.n2_seg = []
        self.ex = dh.patcher(patch_size=self.patch_size)
        slices = self.ex.patchify(path_table=self.df, patient=patient,
                                  mode='classify')
        
        print('there are {} slices'.format(len(slices)))

        for sl in slices:

            print('classifying slices number {}'.format(sl))

            # get patches
            self.ex.patchify(path_table=self.df, patient=patient, mode='classify')
            self.patches = self.ex.patches_xyz

            # predict patches in network 1
            self.classify_network1()

            # find patches that classified as positive in network1 and send to n2
            self.classify_network2()

        coords = []
        for coord in self.n1_seg:
            coords = coords + coord
        self.n1 = np.array(coords)
        coords = []
        for coord in self.n2_seg:
            coords = coords + coord
        self.n2 = np.array(coords)
        
        return [self.n1, self.n2]

