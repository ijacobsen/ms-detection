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
        plt.pause(0.05)
        plt.clf()
    

example_dir = '01016SACH'

pp_data_path = '../preprocessed_data/'
pp_file_name = 'FLAIR_preprocessed.nii'
pp_file = os.path.join(os.path.join(pp_data_path, example_dir), pp_file_name)

tar_data_path = '../raw_data/'
#tar_file_name = 'Consensus.nii'
tar_file_name = 'ManualSegmentation_1.nii'
tar_file = os.path.join(os.path.join(tar_data_path, example_dir), tar_file_name)

pp_img = nib.load(pp_file)
pp_data = pp_img.get_fdata()

tar_img = nib.load(tar_file)
tar_data = tar_img.get_fdata()

animate(pp_data)


#plt.imshow(data[50, :, :], cmap='gray')
