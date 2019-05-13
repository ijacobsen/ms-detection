import os
import pandas as pd
from matplotlib import pyplot as plt
import data_handler as dh
import numpy as np

# load segmentation results
patient = '01016SACH'
filename = patient + '_segmentations.npy'
[n1, n2] = np.load(filename)

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# load consensus
ex = dh.patcher()
con = ex.load_image(path=df.loc[patient]['Consensus'])

# create segmented images
seg_img_n1 = np.zeros(shape=con.shape)
for coord in n1:
    seg_img_n1[tuple(coord[:3].astype(int))] = 1

seg_img_n2 = np.zeros(shape=con.shape)
for coord in n2:
    seg_img_n2[tuple(coord[:3].astype(int))] = 1


