import os
import pandas as pd
from matplotlib import pyplot as plt
import data_handler as dh
import numpy as np
import sys

# the DSC function was written by Sergio Valverde
def DSC(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum




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
    perf.loc[patient]['dice'] = DSC(seg_img_n2, con)
    
    print(perf.loc[patient])

perf.to_pickle('{}{}_performance.pkl'.format(pkl_dir, params))

'''
sl = 40
plt.subplot(2, 1, 1)
plt.imshow(con[sl, :, :])
plt.subplot(2, 1, 2)
plt.imshow(seg_img_n2[sl, :, :])
'''
        

