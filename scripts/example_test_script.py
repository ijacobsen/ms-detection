import model_lib as ml
import data_handler as dh
import os

#%% load data

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# choose patient
patient = dir_list[0]

# load patients data
ex = dh.patcher(mode='testing', patch_size=(11, 11, 11))
ex.patchify(path_table=df, patient=patient)
patches = ex.patches_xyz

 #%% IDK... I DIDNT DO THIS YET
# stack example patches to feed into NN
'''
x_train = [ptch.array for ptch in patches]
xtrain = np.ndarray((len(x_train),
                     x_train[0].shape[0],
                     x_train[0].shape[1],
                     x_train[0].shape[2],
                     num_channels))
'''

#%% load network
# the network tests patches in xyz format... more specifically,
# (batch_size, x, y, z, 1), where 1 is because there is only 1 modality
mdl_dir = 'trained_models_layer1'
model = ml.cnn_model(mode='load', name=patient, path=mdl_dir)


batch_sz = 1
form = np.ndarray((batch_sz, 
                   patches[0].array.shape[0],
                   patches[0].array.shape[1],
                   patches[0].array.shape[2],
                   1))
form[0, :, :, :, 0] = patches[0].array
prediction = model.model.predict(form)

#%% evaluate patient
# for each pixel, create a patch and feed into network

# if positive, store 1, else store 0
