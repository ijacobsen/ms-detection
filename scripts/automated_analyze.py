import os
import pandas as pd
from matplotlib import pyplot as plt
import data_handler as dh
import numpy as np
import sys

# dice score
def dice(tp, fn, fp):
    
    return (100*2*tp)/(fn + fp + 2*tp)

    
    
# parameters
n1lr = sys.argv[1]
n2lr = sys.argv[2]
btchsz = int(sys.argv[3])
params = 'n1lr=' + n1lr + '_n2lr=' + n2lr + '_btch=' + str(btchsz)

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

patient_list = df.index

columns = ['num pixels', 'accuracy', 'false positives', 'false negatives', 'dice']
perf = pd.DataFrame(columns=columns, index=patient_list)

seg_dir = '/scratch/ij405/segments/'
pkl_dir = '/scratch/ij405/pkls/'


for patient in patient_list:
    
    # load segmentation results
    filename = '{}{}_seg_{}.npy'.format(seg_dir, patient, params)
    [n1, n2] = np.load(filename, allow_pickle=True)


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
    true_pos_count = 0
    for coord in mask_coords:
        
        if int(seg_img_n2[coord]) == int(con[coord]):
            correct_count = correct_count + 1
        
        if ((seg_img_n2[coord] == 1) and (con[coord] == 0)):
            false_pos_count = false_pos_count + 1
        
        if ((seg_img_n2[coord] == 0) and (con[coord] == 1)):
            false_neg_count = false_neg_count + 1
        
        if ((seg_img_n2[coord] == 1) and (con[coord] == 1)):
            true_pos_count = true_pos_count + 1
        
    perf.loc[patient]['num pixels'] = len(mask_coords)
    perf.loc[patient]['accuracy'] = float(correct_count)/len(mask_coords)
    perf.loc[patient]['false positives'] = float(false_pos_count)/len(mask_coords)
    perf.loc[patient]['false negatives'] = float(false_neg_count)/len(mask_coords)
    perf.loc[patient]['dice'] = dice(tp=true_pos_count, fn=false_neg_count, fp=false_pos_count)
    np.save('{}img_{}_seg_{}.npy'.format(seg_dir, patient, params), seg_img_n2) 
    print(perf.loc[patient])

# add average row
perf.loc['Average'] = df.mean()

perf.to_pickle('{}{}_performance.pkl'.format(pkl_dir, params))

'''
sl = 40
plt.subplot(2, 1, 1)
plt.imshow(con[sl, :, :])
plt.subplot(2, 1, 2)
plt.imshow(seg_img_n2[sl, :, :])
'''
        

