import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import sys


def animate(data):
    #%matplotlib qt
    for i in range(data[:, 0, 0].shape[0]):
        plt.imshow(data[i, :, :], cmap='gray')
        plt.axis('off')
        plt.title('slide number {}'.format(i))
        plt.show()
        plt.pause(0.02)
        plt.clf()

# fetch data
def fetch_data(data_path, example_dir, file_name):
    file_p = os.path.join(os.path.join(data_path, example_dir), file_name)
    img = (nib.load(file_p)).get_fdata()
    img = (2**16 - 1)*(img - img.min())/(img.max() - img.min())
    return img.astype(np.uint16) # nii files don't have negative values

# fetch data and store in dataframe
def create_df(dir_list):
    
    raw = '../raw_data/'
    prp = '../preprocessed_data/'
    
    modalities = ['DP_preprocessed.nii',
                  'GADO_preprocessed.nii',
                  'T1_preprocessed.nii',
                  'T2_preprocessed.nii',
                  'FLAIR_preprocessed.nii']
    
    img_names = [mod[:-4] for mod in modalities]
    img_names.insert(0, 'Concensus')
    
    # create dataframe
    df = pd.DataFrame(index=dir_list, columns=img_names)
    
    # for each example:
    for dr in dir_list:
    
        # get consensus
        consensus = fetch_data(raw, dr, 'Consensus.nii')
    
        # get all modalities
        data = [fetch_data(prp, dr, mod) for mod in modalities]
        
        # stack into dataframe
        data.insert(0, consensus)

        for mod in modalities:
            df.loc[dr][mod[:-4]] = fetch_data(prp, dr, mod) 
        
        df.loc[dr] = data
    
    #df.to_pickle('~/Projects/MS/scripts/images_dataframe_uint8.pkl')
    
    return df

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [ di for di in dir_list if di[0] == '0']

#df = np.load('images_dataframe_uint8.pkl')

# form database
dir_list = dir_list[0:2]
df = create_df(dir_list)

print(df.head())

