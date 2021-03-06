'''
this script will train 15 CNNs, using leave one out

1.) code pipeline to choose examples for second network

2.) save second network to file

3.) write functions to evaluate performance metrics used in paper (eg dice 
score, etc)

4.) write test script which takes all patches of an image and feeds through 
first network. If first network classifies as non-lesion, classify it as 0. 
else, run patch through second network to get final decision


- all positive examples, same number of negative examples for network 1

- form a new training set which uses all positive examples that were used in 
network 1, as well as the same number of examples which were misclassified 
(i.e. produced a false positive) when tested on network 1 ... must resample 
from universe

'''

import keras
import model_lib as ml
import data_handler as dh
import logger_lib as ll
import numpy as np
import os
import sys

# %%    CONFIGURATION
patch_size = (11, 11, 11) #(x, y, z)
num_channels = 1
epochs_hp = 300
num_pats = 'all'
n1_lr = float(sys.argv[1])
n2_lr = float(sys.argv[2])
batch_sz = int(sys.argv[3])
thresh = float(sys.argv[4])

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# choose a patient
patient_list = list(df.index)
#patient_list = patient_list[:3] # TODO remove this line

log_help = ll.logger(filename='log_btch{}_n1lr={}_n2lr={}_thresh{}'.format(batch_sz, n1_lr, n2_lr, thresh), message='first write')

# %%    CNN training
for k in range(len(patient_list)):

    pats_lv1out = [patient_list[j] for j in range(len(patient_list)) if j != k]

    log_help.update_meta('=========================================')
    log_help.update_meta('=========================================')
    log_help.update_meta('~~~ training network 1 {} ~~~'.format(patient_list[k]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ network 1 prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    first_iteration = True
    for patient in pats_lv1out:
    
        # get patches
        ex = dh.patcher(patch_size=patch_size)
        ex.patchify(path_table=df, patient=patient, mode='network1_train')
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
        # in to account for only 1 channel (modality) being used
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

    log_help.update_meta('training on {} positive examples, {} negative examples'.format(np.sum(ytrain_all[:, 1]), np.sum(ytrain_all[:, 0])))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ network 1 training ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # NOTE: ytrain_all is one-hot ... [0, 1] is a positive example

    # initiate model
    # initiate model
    model_name = '{}_network1_n1lr={}_n2lr={}_btch={}_thresh={}'.format(patient_list[k],
                                                              n1_lr, n2_lr,
                                                              batch_sz, thresh)

    network1 = ml.cnn_model(name=model_name, mode='train', lr=n1_lr)

    # train model
    network1.train_network(xtrain=xtrain_all, ytrain=ytrain_all,
                           batch_size=batch_sz, epochs=epochs_hp)
   
    log_help.update_logger('===========================================')
    log_help.update_logger('===========================================')
    log_help.update_logger('lv1out_network1_{}'.format(patient_list[k])) 
    log_help.update_logger(network1.history)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ network 2 prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    log_help.update_meta('~~~ training network 2 {} ~~~'.format(patient_list[k]))

    # use all positive examples for training layer 2
    pos_examps_idx = np.where(ytrain_all[:, 1] > 0.5)[0]
    xtrain_pos = xtrain_all[pos_examps_idx]
    ytrain_pos = ytrain_all[pos_examps_idx]

    first_iteration = True
    for patient in pats_lv1out:
    
        # get patches
        ex = dh.patcher(patch_size=patch_size)
        ex.patchify(path_table=df, patient=patient, mode='network2_train')
        patches = ex.patches_xyz
    
        # stack example patches to feed into NN
        x_test = [ptch.array for ptch in patches]
        xtest = np.ndarray((len(x_test),
                            x_test[0].shape[0],
                            x_test[0].shape[1],
                            x_test[0].shape[2],
                            num_channels))
        ytest = [int(ptch.label) for ptch in patches]
    
        # TODO this is very sloppy... xtrain[i, :, :, :, 0] ... 0 is hardcoded
        # in to account for only 1 channel (modality) being used
        # fill xdata with patch data
        for i in range(len(xtest)):
            xtest[i, :, :, :, 0] = x_test[i]
        ytest = np.array(ytest)

        # convert target variable into one-hot... [1, 0] for all examples
        y_test = keras.utils.to_categorical(ytest, 2)

        # before stacking, 
        # xtrain (500, 11, 11, 11, 1)
        # y_train (500, 2)

        if first_iteration:
            xtest_all = xtest
            ytest_all = y_test
            first_iteration = False
        else:
            xtest_all = np.concatenate((xtest_all, xtest), 0)
            ytest_all = np.concatenate((ytest_all, y_test), 0)

            # ~~~ JUST GOT TEST PATCHES... FEED THROUGH NETWORK 1,
            # THEN THRESHOLD TO FIND FALSE POSITIVES, THEN STACK TO USE
            # FOR TRAINING IN NETWORK 2 .... 

    # make predictions
    y_predicted = network1.predict_network(xpredict=xtest_all, batch_size=2048)

    # find false positives
    false_pos_truth = y_predicted[:, 1] > thresh
    false_pos_x = xtest_all[false_pos_truth, :, :, :, :]

    # take correct amount of false positives
    #false_pos_x = false_pos_x[:xtrain_pos.shape[0]]
    print('false_pos_x shape is {}'.format(false_pos_x.shape))
    print('xtrain_pos shape is {}'.format(xtrain_pos.shape))

    # take even split of training examples and stack
    if (false_pos_x.shape[0] > xtrain_pos.shape[0]):
        n2_x_train = np.vstack((xtrain_pos, false_pos_x[:xtrain_pos.shape[0], :, :, :, :]))
        #y = np.vstack((np.ones((xtrain_pos.shape[0], 1)),
        #               np.zeros((false_pos_x[:xtrain_pos.shape, :, :, :, :].shape[0], 1))))
    else:
        n2_x_train = np.vstack((xtrain_pos[:false_pos_x.shape[0], :, :, :, :], false_pos_x))
        #y = np.vstack((np.ones((xtrain_pos[:false_pos_x.shape[0], :, :, :, :].shape[0], 1)),
        #               np.zeros((false_pos_x.shape[0], 1))))
    
    # first half are positive examples, second half are negative examples
    y = np.ones((n2_x_train.shape[0], 1))
    y[n2_x_train.shape[0]/2:, :] = np.zeros((n2_x_train.shape[0]/2, 1))

    print('training network 2 on {} patches'.format(y.shape[0]))  

    # convert target variable into one-hot
    n2_y_train = keras.utils.to_categorical(y, 2)

    log_help.update_meta('training on {} positive examples, {} negative examples'.format(np.sum(n2_y_train[:, 1]), np.sum(n2_y_train[:, 0])))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ network 2 training ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # NOTE: ytrain_all is one-hot ... [0, 1] is a positive example

    # initiate model
    # initiate model
    model_name = '{}_network2_n1lr={}_n2lr={}_btch={}_thresh={}'.format(patient_list[k],
                                                              n1_lr, n2_lr,
                                                              batch_sz, thresh)    
    network2 = ml.cnn_model(name=model_name, mode='train', lr=n2_lr)

    # train model
    network2.train_network(xtrain=n2_x_train, ytrain=n2_y_train,
                           batch_size=batch_sz, epochs=epochs_hp)

    log_help.update_logger('===========================================')
    log_help.update_logger('===========================================')
    log_help.update_logger('lv1out_network2_{}'.format(patient_list[k])) 
    log_help.update_logger(network2.history)
# %%    LAYER 2 TRAINING
'''
- form a new training set which uses all positive examples that were used in 
network 1, as well as the same number of examples which were misclassified 
(i.e. produced a false positive) when tested on network 1 ... must resample 
from universe
'''

'''
#%% load network
# the network tests patches in xyz format... more specifically,
# (batch_size, x, y, z, 1), where 1 is because there is only 1 modality
mdl_dir = 'trained_models_layer1'
model = ml.cnn_model(mode='load', name=patient, path=mdl_dir)


batch_sz = 200
form = np.ndarray((batch_sz, 
                   patches[0].array.shape[0],
                   patches[0].array.shape[1],
                   patches[0].array.shape[2],
                   1))
form[0, :, :, :, 0] = patches[0].array
form[1, :, :, :, 0] = patches[1].array

for i in np.arange(batch_sz):
    form[i, :, :, :, 0] = patches[i].array

prediction = model.model.predict(form, batch_size=batch_sz)

print('the prediction is {}'.format(prediction))
'''
