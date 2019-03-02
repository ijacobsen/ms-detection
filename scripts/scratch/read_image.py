import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

def animate(data):
    #%matplotlib qt
    for i in range(data[:, 0, 0].shape[0]):
        plt.imshow(data[i, :, :], cmap='gray')
        plt.axis('off')
        plt.title('slide number {}'.format(i))
        plt.show()
        plt.pause(0.02)
        plt.clf()
    
def subplot_animate(data1, data2, data3=None):
    #%matplotlib qt
    
    if (data3 == None):
        for i in range(data1[:, 0, 0].shape[0]):
            plt.subplot(1, 2, 1)
            plt.imshow(data1[i, :, :], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('slide number {}'.format(i))
            plt.subplot(1, 2, 2)
            plt.imshow(data2[i, :, :], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('slide number {}'.format(i))
            plt.show()
            plt.pause(0.01)
            plt.subplot(1, 2, 1)
            plt.clf()
            plt.subplot(1, 2, 2)
            plt.clf()

    else:        
        for i in range(data1[:, 0, 0].shape[0]):
            plt.subplot(1, 3, 1)
            plt.imshow(data1[i, :, :], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('slide number {}'.format(i))
            plt.subplot(1, 3, 2)
            plt.imshow(data2[i, :, :], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('slide number {}'.format(i))
            plt.subplot(1, 3, 3)
            plt.imshow(data3[i, :, :], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('slide number {}'.format(i))
            plt.show()
            plt.pause(0.01)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.clf()
            plt.subplot(1, 3, 2)
            plt.clf()
            plt.subplot(1, 3, 3)
            plt.clf()



example_dir = '01016SACH'

pp_data_path = '../preprocessed_data/'
pp_file_name = 'FLAIR_preprocessed.nii'
pp_file = os.path.join(os.path.join(pp_data_path, example_dir), pp_file_name)

tar_data_path = '../raw_data/'
tar_file_name = 'Consensus.nii'
#tar_file_name = 'ManualSegmentation_1.nii'
tar_file = os.path.join(os.path.join(tar_data_path, example_dir), tar_file_name)

mask_file_name = 'Mask_registered.nii'
mask_file = os.path.join(os.path.join(pp_data_path, example_dir), mask_file_name)

pp_img = nib.load(pp_file)
pp_data = pp_img.get_data()

tar_img = nib.load(tar_file)
tar_data = tar_img.get_data()

mask_img = nib.load(mask_file)
mask_data = mask_img.get_data()

#animate(tar_data)

#subplot_animate(tar_data, pp_data)
#subplot_animate(tar_data, pp_data, mask_data)

# use mask to find where the brain is... to filter out nonbrain

