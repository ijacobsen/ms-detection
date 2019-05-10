import model_lib as ml
import data_handler as dh
import logger_lib as ll
import numpy as np
import os

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
patient = '01016SACH'
# set directory where models exist
mdl_dir = 'trained_models'

# classify FLAIR image
classifier = ml.classifier(mode='classify', name=patient,
                           path=mdl_dir, data=df)
[n1, n2] = classifier.classify_scan(patient=patient)

# save classified pixels to file
np.save('{}_segmentations'.format(patient), [n1, n2])

# to load
#[n1, n2] = np.load('{}_segmentations.npy'.format(patient))
