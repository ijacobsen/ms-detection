'''
this script will train 


'''

import keras
import model_lib as ml
import data_handler as dh
import logger_lib as ll
import numpy as np
import os

# %%    CONFIGURATION
patch_size = (11, 11, 11) #(x, y, z)
num_channels = 1
batch_sz = 128
epochs_hp = 1000
num_pats = 'all'
n1_lr = 0.003
n2_lr = 0.0003

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

train_pats = ['01016SACH',
              '01038PAGU',
              '01039VITE',
              '07001MOEL',
              '07003SATH',
              '07010NABO',
              '08002CHJE',
              '08027SYBR',
              '08029IVDI']
 
val_pats = ['07040DORE',
            '08031SEVE',
            '01040VANE']
            
test_pats = ['07043SEME',
             '08037ROGU',
             '01042GULE']


# form database
print('loading data')
df = dh.create_df(train_pats + val_pats, modal='flair')
print('data loaded')

patient_list = df.index

log_help = ll.logger(filename='zhi_log_btch{}_p{}_epochs{}_n1lr={}_n2lr={}'.format(batch_sz, len(patient_list), epochs_hp, n1_lr, n2_lr), message='first write')


train_pats

log_help.update_meta('=========================================')
log_help.update_meta('=========================================')
log_help.update_meta('~~~ training network 1 ~~~')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ network 1 prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get training patient data
first_iteration = True
for patient in train_pats:

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

first_iteration = True
for patient in val_pats:

    # get patches
    ex = dh.patcher(patch_size=patch_size)
    ex.patchify(path_table=df, patient=patient, mode='network1_train')
    patches = ex.patches_xyz

    # stack example patches to feed into NN
    x_val = [ptch.array for ptch in patches]
    xval = np.ndarray((len(x_val),
                       x_val[0].shape[0],
                       x_val[0].shape[1],
                       x_val[0].shape[2],
                       num_channels))
    yval = [int(ptch.label) for ptch in patches]

    # TODO this is very sloppy... xtrain[i, :, :, :, 0] ... 0 is hardcoded
    # in to account for only 1 channel (modality) being used
    # fill xdata with patch data
    for i in range(len(xval)):
        xval[i, :, :, :, 0] = x_val[i]
    yval = np.array(yval)

    # convert target variable into one-hot
    y_val = keras.utils.to_categorical(yval, 2)

    # before stacking, 
    # xtrain (500, 11, 11, 11, 1)
    # y_train (500, 2)

    if first_iteration:
        xval_all = xval
        yval_all = y_val
        first_iteration = False
    else:
        xval_all = np.concatenate((xval_all, xval), 0)
        yval_all = np.concatenate((yval_all, y_val), 0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ network 1 training ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NOTE: ytrain_all is one-hot ... [0, 1] is a positive example

# initiate model
model_name = 'zhi_network1'
network1 = ml.cnn_model(name=model_name, mode='train', lr=n1_lr)

# train model
network1.train_network(xtrain=xtrain_all, ytrain=ytrain_all,
                       batch_size=batch_sz, epochs=epochs_hp,
                       val_data=(xval_all, yval_all))
   
log_help.update_logger('===========================================')
log_help.update_logger('===========================================')
log_help.update_logger('zhi_network1') 
log_help.update_logger(network1.history)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ network 2 prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

log_help.update_meta('~~~ training network 2 ~~~')

# use all positive examples for training layer 2
pos_examps_idx = np.where(ytrain_all[:, 1] > 0.5)[0]
xtrain_pos = xtrain_all[pos_examps_idx]
ytrain_pos = ytrain_all[pos_examps_idx]

first_iteration = True
for patient in train_pats:

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
false_pos_truth = y_predicted[:, 1] > 0.7
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

# DUMB
# use all positive examples for training layer 2
pos_examps_idx = np.where(yval_all[:, 1] > 0.5)[0]
xval_pos = xval_all[pos_examps_idx]
yval_pos = yval_all[pos_examps_idx]

first_iteration = True
for patient in val_pats:

    # get patches
    ex = dh.patcher(patch_size=patch_size)
    ex.patchify(path_table=df, patient=patient, mode='network2_train')
    patches = ex.patches_xyz

    # stack example patches to feed into NN
    x_val = [ptch.array for ptch in patches]
    xval = np.ndarray((len(x_val),
                       x_val[0].shape[0],
                       x_val[0].shape[1],
                       x_val[0].shape[2],
                       num_channels))
    yval = [int(ptch.label) for ptch in patches]

    # TODO this is very sloppy... xtrain[i, :, :, :, 0] ... 0 is hardcoded
    # in to account for only 1 channel (modality) being used
    # fill xdata with patch data
    for i in range(len(xval)):
        xval[i, :, :, :, 0] = x_val[i]
    yval = np.array(yval)

    # convert target variable into one-hot... [1, 0] for all examples
    y_val = keras.utils.to_categorical(yval, 2)

    # before stacking, 
    # xtrain (500, 11, 11, 11, 1)
    # y_train (500, 2)

    if first_iteration:
        xval_all = xval
        yval_all = y_val
        first_iteration = False
    else:
        xval_all = np.concatenate((xval_all, xval), 0)
        yval_all = np.concatenate((yval_all, y_val), 0)

        # ~~~ JUST GOT TEST PATCHES... FEED THROUGH NETWORK 1,
        # THEN THRESHOLD TO FIND FALSE POSITIVES, THEN STACK TO USE
        # FOR TRAINING IN NETWORK 2 .... 

# make predictions
y_predicted = network1.predict_network(xpredict=xval_all, batch_size=2048)

# find false positives
false_pos_truth = y_predicted[:, 1] > 0.7
false_pos_x = xval_all[false_pos_truth, :, :, :, :]

# take correct amount of false positives
#false_pos_x = false_pos_x[:xtrain_pos.shape[0]]
print('false_pos_x validation shape is {}'.format(false_pos_x.shape))
print('xval_pos validation shape is {}'.format(xval_pos.shape))

# take even split of training examples and stack
if (false_pos_x.shape[0] > xval_pos.shape[0]):
    n2_x_val = np.vstack((xval_pos, false_pos_x[:xval_pos.shape[0], :, :, :, :]))
    #y = np.vstack((np.ones((xtrain_pos.shape[0], 1)),
    #               np.zeros((false_pos_x[:xtrain_pos.shape, :, :, :, :].shape[0], 1))))
else:
    n2_x_val = np.vstack((xval_pos[:false_pos_x.shape[0], :, :, :, :], false_pos_x))
    #y = np.vstack((np.ones((xtrain_pos[:false_pos_x.shape[0], :, :, :, :].shape[0], 1)),
    #               np.zeros((false_pos_x.shape[0], 1))))

# first half are positive examples, second half are negative examples
y = np.ones((n2_x_val.shape[0], 1))
y[n2_x_val.shape[0]/2:, :] = np.zeros((n2_x_val.shape[0]/2, 1)) 
# convert target variable into one-hot
n2_y_val = keras.utils.to_categorical(y, 2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ network 2 training ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NOTE: ytrain_all is one-hot ... [0, 1] is a positive example

# initiate model
model_name = 'zhi_network2'
network2 = ml.cnn_model(name=model_name, mode='train', lr=n2_lr)

# train model
network2.train_network(xtrain=n2_x_train, ytrain=n2_y_train,
                       batch_size=batch_sz, epochs=epochs_hp,
                       val_data=(n2_x_val, n2_y_val))

log_help.update_logger('===========================================')
log_help.update_logger('===========================================')
log_help.update_logger('zhi_network2')
log_help.update_logger(network2.history)
