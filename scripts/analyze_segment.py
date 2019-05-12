import os
import pandas as pd
from matplotlib import pyplot as plt
import data_handler as dh
import numpy as np

# load segmentation results
patient = '01016SACH'
filename = patient + '_segmentations.npy'
[n1, n2] = np.load(filename)



seg_coords = []
for sl in seg_results:
    seg_coords.append(sl)

seg_coords = list(list(list(list(seg_results))))

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

