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
def fetch_data(data_path, example_dir, file_name, quant=True):
    file_p = os.path.join(os.path.join(data_path, example_dir), file_name)
    img = (nib.load(file_p)).get_data()
    img = (2**16 - 1)*(img - img.min())/(img.max() - img.min())
    if quant:
        return img.astype(np.uint16) # nii files don't have negative values
    else:
        return img

# fetch data and store in dataframe
def create_df(dir_list, quant=True):
    
    raw = '../raw_data/'
    prp = '../preprocessed_data/'
    
    #modalities = ['FLAIR_preprocessed.nii.gz']

    modalities = ['DP_preprocessed.nii.gz',
                  'GADO_preprocessed.nii.gz',
                  'T1_preprocessed.nii.gz',
                  'T2_preprocessed.nii.gz',
                  'FLAIR_preprocessed.nii.gz']

    
    img_names = [mod[:-4] for mod in modalities]
    img_names.insert(0, 'Concensus')
    
    # create dataframe
    df = pd.DataFrame(index=dir_list, columns=img_names)
    
    # for each example:
    for dr in dir_list:
    
        # get consensus
        consensus = fetch_data(raw, dr, 'Consensus.nii.gz')
    
        # get all modalities
        data = [fetch_data(prp, dr, mod, quant) for mod in modalities]
        
        # stack consensus
        data.insert(0, consensus)
        
        # stack into dataframe
        df.loc[dr] = data
    
    df.to_pickle('./images_dataframe_uint16.pkl')
    df.to_hdf('./images_dataframe_uint16.h5', 'table', append=True)    

    return df

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [ di for di in dir_list if di[0] == '0']

#df = np.load('images_dataframe_uint8.pkl')

# form database
dir_list = dir_list[0:2]
#[data, df] = create_df(dir_list)
print('loading data')
df = create_df(dir_list)
print('data loaded')

#df_full = create_df(dir_list, quant=False)

#print(df.head())

