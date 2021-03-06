import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
from random import shuffle
from matplotlib.patches import Circle


def show_patch(ptch, con_ptch=False, animate=False):

    if (type(con_ptch) == np.ndarray):
        # show_patch(patches[12], con_patches[12])
        print('plotting consensus and flair patch')
        center = ptch.array.shape[0]/2
        plt.subplot(1, 2, 1)
        plt.imshow(ptch.array[center, :, :], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(con_ptch[center, :, :], cmap='gray')


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

    def __init__(self, patch_size=(11, 11, 11)):

        self.patch_size = patch_size  # (x, y, slice)
        self.consensus = 0
        self.flair = 0
        self.mask = 0
        self.patches = 0
        self.patches_xyz = 0
        self.consensus_patches = 0
        self.classify_iter_number = 0
        self.classify_meta_data_prep = True

    def load_image(self, path=False, highres=False, normalize=False):

        if (path):
            img = (nib.load(path)).get_data()
        else:
            print('----no path given----')
            return False

        if (normalize):
            img = (img - img.mean()) / img.var()
            #img = (2**8 - 1)*(img - img.min())/(img.max() - img.min())
        if (not highres):
            #return img.astype(np.uint8)  # nii files don't have negative values
            return img
        else:
            return img

    def get_patches(self, img, coords, num_patches='test'):

        if (num_patches == 'test'):
            num_patches = len(coords)

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

    def patchify(self, path_table, patient, num_patches=30000,
                 modals=False, mode='modeless'):

        ## ADD ~~TESTING~~ PATCHIFY AT A LATER POINT TODO

        # use the mask to filter out black space
        mask = self.load_image(path=path_table.loc[patient]['Mask'],
                               normalize=False)
        self.mask = mask

        # coordinates where the brain is present
        valid_coords = tuple(zip(*(np.nonzero(mask))))
        valid_coords = list(valid_coords)
        shuffle(valid_coords)

        if (not modals):

            # load FLAIR image
            flair = self.load_image(path=path_table.loc[patient]['FLAIR_preprocessed'], 
                                    normalize=True)
            self.flair = flair

        else:

            print('we only support accessing FLAIR right now !')

        # if training, filter out similar pixels
        if (mode == 'network1_train'):

            # load the consensus
            con = self.load_image(path=path_table.loc[patient]['Consensus'])
            self.consensus = con
            pos_coords = tuple(zip(*(np.nonzero(con))))

            # we have a lot of positive examples, so lets set a minimum
            # distance between coordinates to use in training
            # WAS (1, 4, 4)
            min_dist = (1, 4, 4)

            # downsample because coords are ordered, and below code is O(n^2)
            # WAS downsampled by 20
            ds_pos_coords = pos_coords[::1]
            
            
            '''
            this is a major bootleneck in the code
            this is the correct way to do it, but for the sake of 
            a quicker runtime i will just downsample
            # finds good candidates... slow O(n^2) in ds_pos_coords
            pos_used = [pos_coords[0], pos_coords[-1]]
            [pos_used.append(coord) for coord in ds_pos_coords if (np.apply_along_axis(np.any, 1, 
             np.abs(np.array(coord) - np.array(pos_used)) > np.array(min_dist)).all())]
            '''

            # actually i will hash using an image
            pos_used = [pos_coords[0]]
            used_img = 0*np.copy(self.consensus)
            used_img[pos_used[0]] = 1
            for coord in ds_pos_coords:
                
                # make sure coordinate is in bounds
                if ((coord[0]<used_img.shape[0]) and 
                    (coord[1]<used_img.shape[1]) and 
                    (coord[2]<used_img.shape[2])):
                    
                    if (np.max(used_img[coord[0]-min_dist[0]:coord[0]+min_dist[0],
                                        coord[1]-min_dist[1]:coord[1]+min_dist[1],
                                        coord[2]-min_dist[2]:coord[2]+min_dist[2]]) < 0.5):                      
                        used_img[coord] = 1
            
            # positive coordinates to use
            pos_used = list(zip(*(np.nonzero(used_img))))

            # shuffle coordinates
            shuffle(pos_used)

            # get positive patches
            pos_patches = self.get_patches(img=self.flair, num_patches=len(pos_used),
                                           coords=pos_used)

            # assigned labels to patches
            for ptch in pos_patches:
                ptch.label = '1'
            num_pos_returned = len(pos_patches)

            # locate negative patches
            contenders_idx = np.random.randint(0, len(valid_coords),
                                               3*num_pos_returned)
            contenders = [valid_coords[idx] for idx in contenders_idx]
            neg_used = [i for i in contenders if i not in pos_used]
            
            # shuffle coordinates
            shuffle(neg_used)

            # get negative patches and assign labels
            neg_patches = self.get_patches(img=self.flair, num_patches=num_pos_returned,
                                           coords=neg_used)
            for ptch in neg_patches:
                ptch.label = '0'
            num_neg_returned = len(neg_patches)

            '''
            if (len(pos_patches) > num_patches/2):
                patches_to_return = pos_patches[:num_patches/2] + neg_patches
                num_pos_returned = num_patches/2
            else:
                patches_to_return = pos_patches + neg_patches
                num_pos_returned = len(pos_patches)
            '''
            patches_to_return = pos_patches + neg_patches

            print('returning {} positive patches, {} negative patches'.format(num_pos_returned,
                  num_neg_returned))

            #patches = patches_to_return[:num_patches]
            patches = patches_to_return[:]
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

        elif (mode == 'network2_train'):

            print('preparing network 2 training patches')

            # load the consensus
            con = self.load_image(path=path_table.loc[patient]['Consensus'])
            self.consensus = con
            pos_coords = tuple(zip(*(np.nonzero(self.consensus))))

            # i am fetching mask coordinates because all coordinates
            # are way too many... 40 million... after downsampling, 80,000
            # get coordinates
            # sloppy way to get all coordinates, but it works
            # flair_coords = tuple(zip(*(np.where(self.mask >= 0))))

            # downsampling mask to get a smaller number of patches
            ds_factor = [1, 4, 4]
            eval_coords = np.array((zip(*(np.nonzero(self.mask[::ds_factor[0],
                                                               ::ds_factor[1],
                                                               ::ds_factor[2]])))))
            eval_coords = eval_coords * np.array([ds_factor[0],
                                                  ds_factor[1],
                                                  ds_factor[2]])

            # filter out positive examples... we only want negative examples
            neg_coords = []
            for eval_co in eval_coords:
                if not self.consensus[tuple(eval_co)]:
                    neg_coords.append(tuple(eval_co))

            # shuffle negative examples
            shuffle(neg_coords)

            '''
            for debugging... to verify eval_coords are in mask
            mask_coords = np.array((zip(*(np.nonzero(self.mask)))))
            check = 0
            for eval_co in eval_coords:
                if eval_co in mask_coords:
                    check = check + 1
            if len(eval_co) != check ... then theres a problem
            '''

            print('fetching {} patches'.format(len(neg_coords)))

            # get patches... based on mask_coords
            patches = self.get_patches(img=self.flair, 
                                       num_patches='test',
                                       coords=neg_coords)
            '''
            below block was used for troubleshooting
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

            # we only have negative examples in network2_training
            for ptch in patches:
                ptch.label = '0'

            self.patches = patches

            # reshape patches to meet (x, y, z) criteria
            self.patches_xyz = patches
            for i in np.arange(len(self.patches_xyz)):
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 0, 1)
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 1, 2)
                
            return 0

        elif (mode == 'classify'):
            
            # if its the first iteration, do some prep
            if (self.classify_meta_data_prep == True):

                print('preparing image')

                # i am fetching mask coordinates because all coordinates
                # are way too many... 40 million... to save memory on patches
                # i use the mask
                # flair_coords = tuple(zip(*(np.where(self.mask >= 0))))
                mask_coords = tuple(zip(*(np.nonzero(self.mask))))
                print('{} pixels in the FLAIR will be classified'.format(len(mask_coords)))
                
                # unique list of valid slices
                slices = [coord[0] for coord in mask_coords]
                self.slices = list(set(slices))
                
                self.classify_meta_data_prep = False
                
                return self.slices

            else:
                
                # if not first iteration, increment slice index number
                slice_idx = self.slices[self.classify_iter_number]


            print('getting patches for slice number {}'.format(slice_idx))

            # get coordinates in current slice
            coords = tuple(zip(*(np.nonzero(self.mask[slice_idx, :, :]))))
            coords = [[slice_idx] + list(coord) for coord in coords]

            # get patches... based on mask_coords
            patches = self.get_patches(img=self.flair, 
                                       num_patches='test',
                                       coords=coords)
            self.patches = patches

            # reshape patches to meet (x, y, z) criteria
            self.patches_xyz = patches
            for i in np.arange(len(self.patches_xyz)):
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 0, 1)
                self.patches_xyz[i].array = np.moveaxis(self.patches_xyz[i].array, 1, 2)
            
            # increase classify iteration index
            self.classify_iter_number = self.classify_iter_number + 1
                
            return 0
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

