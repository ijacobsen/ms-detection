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
        plt.pause(0.25)
        plt.clf()

# fetch data and store in dataframe
def create_df(dir_list, quant=True, modal='all'):
    
    raw = '../raw_data/'
    prp = '../preprocessed_data/'

    if (modal == 'all'):
        modalities = ['DP_preprocessed.nii.gz',
                      'GADO_preprocessed.nii.gz',
                      'T1_preprocessed.nii.gz',
                      'T2_preprocessed.nii.gz',
                      'FLAIR_preprocessed.nii.gz']

    else:
        modalities = ['FLAIR_preprocessed.nii.gz']

    img_names = [mod[:-7] for mod in modalities]
    img_names.insert(0, 'Consensus')
    img_names.insert(1, 'Mask')
    
    # create dataframe
    df = pd.DataFrame(index=dir_list, columns=img_names)
    
    # for each example:
    for dr in dir_list:

        # get consensus
        consensus = os.path.join(os.path.join(raw, dr), 'Consensus.nii.gz')

        # get mask
        mask = os.path.join(os.path.join(prp, dr), 'Mask_registered.nii.gz')

        # get all modalities
        data = [os.path.join(os.path.join(prp, dr), mod) for mod in modalities]

        # stack consensus
        data.insert(0, consensus)
        data.insert(1, mask)

        # stack into dataframe
        df.loc[dr] = data 

    return df

class patch(object):
    
    def __init__(self):
        self.coords = np.nan
        self.array = np.nan
        self.label = np.nan


class patcher(object):

    def __init__(self, patch_size=(11, 11, 11)):

        self.patch_size = patch_size

    def load_image(self, path=False, highres=False, normalize=False):

        if (path):
            img = (nib.load(path)).get_data()
        else:
            print('----no path given----')
            return False

        if (normalize):
            img = (2**8 - 1)*(img - img.min())/(img.max() - img.min())
        if (not highres):
            return img.astype(np.uint8) # nii files don't have negative values
        else:
            return img

    def get_patches(self, img, coords, num_patches=500):

        img_shape = img.shape

        np.random.shuffle(coords)
        patch_list = []
        i = 0
        for coord in coords:

            sl_min = coord[0] - self.patch_size[0]/2
            sl_max = coord[0] + self.patch_size[0]/2 + 1

            x_min = coord[1] - self.patch_size[1]/2
            x_max = coord[1] + self.patch_size[1]/2 + 1

            y_min = coord[2] - self.patch_size[2]/2
            y_max = coord[2] + self.patch_size[2]/2 + 1
            
            # make sure slice indices are valid
            if ((np.array([sl_min, x_min, y_min]) > 0).all() and 
                (np.array([sl_max, x_max, y_max]) < np.array(img_shape)).all()):

                new_patch = patch()
                new_patch.array = img[sl_min:sl_max, x_min:x_max, y_min:y_max]
                new_patch.label = 'unknown'
                new_patch.coords = coord
                patch_list.append(new_patch)
                i = i + 1

            if (i >= num_patches):
                break

        return patch_list

    def patchify(self, path_table, patient, num_patches=500,
                  modals=False, training=True, testing=False):

        ## ADD ~~TESTING~~ PATCHIFY AT A LATER POINT TODO

        # use the mask to filter out black space
        mask = self.load_image(path=path_table.loc[patient]['Mask'])
        img_size = mask.shape

        # coordinates where the brain is present
        valid_coords = tuple(zip(*(np.nonzero(mask))))

        if (not modals):

            flair = self.load_image(path=path_table.loc[patient]['FLAIR_preprocessed'], 
                                    normalize=True)

        else:

            print('we only support accessing FLAIR right now !')

        # if training, filter out similar pixels
        if (training):

            # load the consensus
            con = self.load_image(path=path_table.loc[patient]['Consensus'])
            pos_coords = tuple(zip(*(np.nonzero(con))))

            # we have a lot of positive examples, so lets set a minimum
            # distance between coordinates to use in training
            min_dist = (2, 20, 20)

            # downsample because coords are ordered, and below code is O(n^2) 
            ds_pos_coords = pos_coords[::40]

            # finds good candidates... slow O(n^2) in ds_pos_coords
            pos_used = [pos_coords[0], pos_coords[-1]]
            [pos_used.append(coord) for coord in ds_pos_coords if (np.apply_along_axis(np.any, 1, 
             np.abs(np.array(coord) - np.array(pos_used)) > np.array(min_dist)).all())]

            # get patches and assign labels
            pos_patches = self.get_patches(img=flair, num_patches=num_patches, coords=pos_used)
            for ptch in pos_patches: ptch.label='1'

            # locate negative patches
            contenders_idx = np.random.randint(0, len(valid_coords), 2*num_patches)
            contenders = [valid_coords[idx] for idx in contenders_idx]
            neg_used = [i for i in contenders if i not in pos_used]

            # get negative patches and assign labels
            neg_patches = self.get_patches(img=flair, num_patches=num_patches, coords=neg_used)
            for ptch in neg_patches: ptch.label='0'
            
            if (len(pos_patches) > num_patches/2):
                patches_to_return = pos_patches[:num_patches/2] + neg_patches
                num_pos_returned = num_patches/2
            else:
                patches_to_return = pos_patches + neg_patches  
                num_pos_returned = len(pos_patches)
            print('returning {} positive patches, {} negative patches'.format(num_pos_returned,
                  num_patches-num_pos_returned))
            
            return patches_to_return[:num_patches]


# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [ di for di in dir_list if di[0] == '0']


# form database
print('loading data')
df = create_df(dir_list, modal='flair') 
print('data loaded')

# choose a patient
patient_list = df.index
patient = patient_list[0]

# get patches
ex = patcher(patch_size = (21, 35, 35))
patches = ex.patchify(path_table=df, patient=patient)

#print(df.head())

