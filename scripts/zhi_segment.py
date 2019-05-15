import model_lib as ml
import data_handler as dh
import logger_lib as ll
import numpy as np
import os

# TODO coontinue here... fix zhi model names so load_Model works correctly

# select hyperparameters
n1lr = '0.03'
n2lr = '0.003'
btchsz = 128
epochs = 1000

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# test patients
test_pats = ['07043SEME',
             '08037ROGU',
             '01042GULE']

# set directory where models exist
mdl_dir = '/scratch/ij405/zhi_models'

# segment image
for patient in test_pats:

    # classify FLAIR image
    classifier = ml.classifier(mode='classify', name=patient,
                               path=mdl_dir, data=df)
    [n1, n2] = classifier.classify_scan(patient=patient)
    
    # save classified pixels to file
    np.save('{}_seg_'.format(patient), [n1, n2])
    np.savez_compressed('{}_segmentations_compressed'.format(patient), [n1, n2])

# to load
#[n1, n2] = np.load('{}_segmentations.npy'.format(patient))
