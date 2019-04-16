import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.patches import Circle


def show_patch(ptch, con_ptch=False, animate=False):

    if con_ptch:
        # show_patch(patches[12], con_patches[12])
        center = ptch.array.shape[0]/2
        plt.subplot(1, 2, 1)
        plt.imshow(ptch.array[center, :, :], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(con_ptch.array[center, :, :], cmap='gray')


    elif not animate:
        # show_patch(patches[3])
        slide_center = ptch.array.shape[0]/2
        center = (ptch.array.shape[1]/2, ptch.array.shape[2]/2)
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(ptch.array[slide_center, :, :], cmap='gray')
        circ1 = Circle(center, 3, facecolor='None', edgecolor='r',
                       lw=2, zorder=10)
        ax.add_patch(circ1)

    elif animate:
        # show_patch(patches, animate=True)
        slide_center = ptch[0].array.shape[0]/2
        center = (ptch[0].array.shape[1]/2, ptch[0].array.shape[2]/2)
        patches_to_show = ptch[:10] + ptch[-10:]
        fig, ax = plt.subplots(1)
        for sl in patches_to_show:
            # fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            ax.imshow(sl.array[slide_center, :, :], cmap='gray')
            if sl.label == '1':
                circ1 = Circle(center, 3, facecolor='None', edgecolor='r',
                               lw=2, zorder=10)
            else:
                circ1 = Circle(center, 3, facecolor='None', edgecolor='g',
                               lw=2, zorder=10)
            ax.add_patch(circ1)
            ax.set_title(sl.label)
            plt.pause(0.8)
            plt.cla()


def show_patch_xyz(ptch, con_ptch=False, animate=False):

    if con_ptch:
        # show_patch(patches[12], con_patches[12])
        center = ptch.array.shape[2]/2
        plt.subplot(1, 2, 1)
        plt.imshow(ptch.array[:, :, center], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(con_ptch.array[:, :, center], cmap='gray')

    elif not animate:
        # show_patch(patches[3])
        slide_center = ptch.array.shape[2]/2
        center = (ptch.array.shape[0]/2, ptch.array.shape[1]/2)
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(ptch.array[:, :, slide_center], cmap='gray')
        circ1 = Circle(center, 3, facecolor='None', edgecolor='r',
                       lw=2, zorder=10)
        ax.add_patch(circ1)

    elif animate:
        # show_patch(patches, animate=True)
        slide_center = ptch[0].array.shape[2]/2
        center = (ptch[0].array.shape[0]/2, ptch[0].array.shape[1]/2)
        patches_to_show = ptch[:10] + ptch[-10:]
        fig, ax = plt.subplots(1)
        for sl in patches_to_show:
            # fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            ax.imshow(sl.array[:, :, slide_center], cmap='gray')
            if sl.label == '1':
                circ1 = Circle(center, 3, facecolor='None', edgecolor='r',
                               lw=2, zorder=10)
            else:
                circ1 = Circle(center, 3, facecolor='None', edgecolor='g',
                               lw=2, zorder=10)
            ax.add_patch(circ1)
            ax.set_title(sl.label)
            plt.pause(0.8)
            plt.cla()


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

    def __init__(self, mode='training', patch_size=(11, 11, 11)):

        self.mode = mode
        self.patch_size = patch_size  # (x, y, slice)
        self.consensus = 0
        self.flair = 0
        self.mask = 0
        self.patches = 0
        self.patches_xyz = 0
        self.consensus_patches = 0

    def load_image(self, path=False, highres=False, normalize=False):

        if (path):
            img = (nib.load(path)).get_data()
        else:
            print('----no path given----')
            return False

        if (normalize):
            img = (2**8 - 1)*(img - img.min())/(img.max() - img.min())
        if (not highres):
            return img.astype(np.uint8)  # nii files don't have negative values
        else:
            return img

    def get_patches(self, img, coords, num_patches='test'):

        if (num_patches == 'test'):
            num_patches = 10
            # TODO switch back num_patches = len(coords)

        img_shape = img.shape

        # np.random.shuffle(coords)
        patch_list = []
        i = 0
        for coord in coords:

            sl_min = coord[0] - self.patch_size[2]/2
            sl_max = coord[0] + self.patch_size[2]/2 + 1

            # print('min {}, max{}'.format(sl_min, sl_max))

            x_min = coord[1] - self.patch_size[0]/2
            x_max = coord[1] + self.patch_size[0]/2 + 1

            y_min = coord[2] - self.patch_size[1]/2
            y_max = coord[2] + self.patch_size[1]/2 + 1

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

    def patchify(self, path_table, patient, num_patches=300,
                 modals=False):

        ## ADD ~~TESTING~~ PATCHIFY AT A LATER POINT TODO

        # use the mask to filter out black space
        mask = self.load_image(path=path_table.loc[patient]['Mask'])
        self.mask = mask

        # coordinates where the brain is present
        valid_coords = tuple(zip(*(np.nonzero(mask))))

        if (not modals):

            flair = self.load_image(path=path_table.loc[patient]['FLAIR_preprocessed'], 
                                    normalize=True)
            self.flair = flair

        else:

            print('we only support accessing FLAIR right now !')

        # if training, filter out similar pixels
        if (self.mode == 'training'):

            # load the consensus
            con = self.load_image(path=path_table.loc[patient]['Consensus'])
            self.consensus = con
            pos_coords = tuple(zip(*(np.nonzero(con))))

            # we have a lot of positive examples, so lets set a minimum
            # distance between coordinates to use in training
            min_dist = (2, 6, 6)

            # downsample because coords are ordered, and below code is O(n^2)
            ds_pos_coords = pos_coords[::30]

            # finds good candidates... slow O(n^2) in ds_pos_coords
            pos_used = [pos_coords[0], pos_coords[-1]]
            [pos_used.append(coord) for coord in ds_pos_coords if (np.apply_along_axis(np.any, 1, 
             np.abs(np.array(coord) - np.array(pos_used)) > np.array(min_dist)).all())]

            # get patches
            pos_patches = self.get_patches(img=self.flair, num_patches=num_patches,
                                           coords=pos_used)

            # assigned labels to patches
            for ptch in pos_patches:
                ptch.label = '1'

            # locate negative patches
            contenders_idx = np.random.randint(0, len(valid_coords),
                                               2*num_patches)
            contenders = [valid_coords[idx] for idx in contenders_idx]
            neg_used = [i for i in contenders if i not in pos_used]

            # get negative patches and assign labels
            neg_patches = self.get_patches(img=self.flair, num_patches=num_patches,
                                           coords=neg_used)
            for ptch in neg_patches:
                ptch.label = '0'

            if (len(pos_patches) > num_patches/2):
                patches_to_return = pos_patches[:num_patches/2] + neg_patches
                num_pos_returned = num_patches/2
            else:
                patches_to_return = pos_patches + neg_patches
                num_pos_returned = len(pos_patches)
            print('returning {} positive patches, {} negative patches'.format(num_pos_returned,
                  num_patches-num_pos_returned))

            patches = patches_to_return[:num_patches]
            self.patches = patches
            
            # reshape patches to meet (x, y, z) criteria
            self.patches_xyz = patches
            for i in np.arange(len(self.patches_xyz)):
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 0, 1)
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 1, 2)
            '''
            if debug:
                print('warning: DEBUG IS ON!')
                coordinates = [ptch.coords for ptch in patches]
                self.consensus_patches = self.get_patches(img=con, 
                                                          coords=coordinates)
                print('debug consensus patches fetched')
            '''

            return 0

        elif (self.mode == 'testing'):

            print('preparing test image')

            # load the consensus
            con = self.load_image(path=path_table.loc[patient]['Consensus'])
            self.consensus = con
            pos_coords = tuple(zip(*(np.nonzero(self.consensus))))

            # i am fetching mask coordinates because all coordinates
            # are way too many... 40 million
            # get coordinates
            # sloppy way to get all coordinates, but it works
            # flair_coords = tuple(zip(*(np.where(self.mask >= 0))))
            mask_coords = tuple(zip(*(np.nonzero(self.mask))))

            print('fetching {} patches'.format(len(mask_coords)))

            # get patches... based on mask_coords
            patches = self.get_patches(img=self.flair, 
                                       num_patches='test',
                                       coords=mask_coords)
            '''
            patches = ex.get_patches(img=ex.flair, 
                                       num_patches='test',
                                       coords=mask_coords)

            for ptch in patches:
                if ex.consensus[ptch.coords]:
                    ptch.label = '1'
                else:
                    ptch.label = '0'

            check = 0
            for ptch in patches: check = check + int(ptch.label)
            '''

            for ptch in patches:
                if self.consensus[ptch.coords]:
                    ptch.label = '1'
                else:
                    ptch.label = '0'

            self.patches = patches

            # reshape patches to meet (x, y, z) criteria
            self.patches_xyz = patches
            for i in np.arange(len(self.patches_xyz)):
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 0, 1)
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 1, 2)
                
            return 0
            
'''
debug = False

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
ex = patcher(patch_size = (25, 25, 11))
ex.patchify(path_table=df, patient=patient)
con_patches = ex.consensus_patches
patches = ex.patches_xyz

#show_patch_xyz(patches, animate=True)
#show_patch_xyz(patches[12], con_patches[12])
#show_patch_xyz(patches[3])

'''

