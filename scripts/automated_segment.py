import model_lib as ml
import data_handler as dh
import logger_lib as ll
import numpy as np
import os
import sys

# select hyperparameters
n1lr = sys.argv[1]
n2lr = sys.argv[2]
btchsz = int(sys.argv[3])

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# test patients
patient_list = df.index

# set directory where models exist
mdl_dir = '/scratch/ij405/models'
seg_dir = '/scratch/ij405/segments/'

# parameters
n1params = 'lr=' + n1lr + 'btch=' + str(btchsz)
n2params = 'lr=' + n2lr + 'btch=' + str(btchsz)
params = 'n1lr=' + n1lr + '_n2lr=' + n2lr + '_btch=' + str(btchsz) 

# segment image
for patient in patient_list:
    
    # classify FLAIR image
    classifier = ml.classifier(mode='classify', patient=patient,
                               n1name=params, n2name=params,
                               path=mdl_dir, data=df, zhi=False)
    [n1, n2] = classifier.classify_scan(patient=patient)
    
    # save classified pixels to file
    np.save('{}{}_seg_{}'.format(seg_dir, patient, params), [n1, n2])
    #np.savez_compressed('{}_seg_compressed_{}'.format(patient, params), [n1, n2])

# to load
#[n1, n2] = np.load('{}_segmentations.npy'.format(patient))
