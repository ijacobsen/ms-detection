import os
import pandas as pd
from matplotlib import pyplot as plt
import data_handler as dh
import numpy as np

# parameters
btchsz = 32
n1lr = '0.03'
n2lr = '0.003'
params = 'n1lr=' + n1lr + '_n2lr=' + n2lr + '_btch=' + str(btchsz)

# test patients
test_pats = ['07043SEME',
             '08037ROGU',
             '01042GULE']

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

columns = ['num pixels', 'accuracy', 'false positives', 'false negatives']
perf = pd.DataFrame(columns=columns, index=test_pats)

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

for patient in test_pats:
    
    # load segmentation results
    filename = '{}_seg_{}.npy'.format(patient, params)
    [n1, n2] = np.load(filename)


    # load consensus
    ex = dh.patcher()
    con = ex.load_image(path=df.loc[patient]['Consensus'])
    mask = ex.load_image(path=df.loc[patient]['Mask'])

    # create segmented images
    seg_img_n1 = np.zeros(shape=con.shape)
    for coord in n1:
        seg_img_n1[tuple(coord[:3].astype(int))] = 1

    seg_img_n2 = np.zeros(shape=con.shape)
    for coord in n2:
        seg_img_n2[tuple(coord[:3].astype(int))] = 1

    # find patches that were classified
    mask_coords = tuple(zip(*(np.nonzero(mask))))

    correct_count = 0
    false_pos_count = 0
    false_neg_count = 0
    for coord in mask_coords:
        
        if int(seg_img_n2[coord]) == int(con[coord]):
            correct_count = correct_count + 1
        
        if ((seg_img_n2[coord] == 1) and (con[coord] == 0)):
            false_pos_count = false_pos_count + 1
        
        if ((seg_img_n2[coord] == 0) and (con[coord] == 1)):
            false_neg_count = false_neg_count + 1
        
    perf.loc[patient]['num pixels'] = len(mask_coords)
    perf.loc[patient]['accuracy'] = float(correct_count)/len(mask_coords)
    perf.loc[patient]['false positives'] = float(false_pos_count)/len(mask_coords)
    perf.loc[patient]['false negatives'] = float(false_neg_count)/len(mask_coords)
    
    print(perf.loc[patient])

perf.to_pickle('{}_performance.pkl'.format(params))

'''
sl = 40
plt.subplot(2, 1, 1)
plt.imshow(con[sl, :, :])
plt.subplot(2, 1, 2)
plt.imshow(seg_img_n2[sl, :, :])
'''
        

