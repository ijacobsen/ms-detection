import keras
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

mdl_dir = 'trained_models'
classifier = ml.classifier(mode='classify', name=patient,
                           path=mdl_dir, data=df)
p
[n1, n2] = classifier.classify_scan(patient=patient)