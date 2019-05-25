import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import data_handler as dh
import os

n1lr = '0.3'
n2lr = '0.3'
b = '128'
patient = '01016SACH'
dir_name = '/Users/ian/Projects/MS/new_final/'
sl = [50, 55, 60, 70, 80, 90]

#%%

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# load consensus
ex = dh.patcher()
con = ex.load_image(path=df.loc[patient]['Consensus'])
mask = ex.load_image(path=df.loc[patient]['Mask'])


#%% make latex table and stem plot

n1lr = '0.0003'
n2lr = '0.0003'
b = '512'

filename = 'n1lr={}_n2lr={}_btch={}_performance.pkl'.format(n1lr, n2lr, b)
df = np.load(os.path.join(dir_name, filename))
df.loc['Average'] = df.mean(skipna=True)

print(df.to_latex())

plt.title('n1lr={}, n2lr={}, b={}'.format(n1lr, n2lr, b))
plt.stem(df.index, np.array(df['dice']))
plt.xticks(rotation=90)
plt.ylim([0, 65])
plt.ylabel('Dice Score')
plt.savefig('stem_{}_{}_{}.png'.format(n1lr, n2lr, b), dpi=100)

#%%

patient = '08029IVDI'
n1lr = '0.03'
n2lr = '0.03'
b = '512'
sl = [40, 45, 50, 55, 60, 70, 80, 90]

filename = 'img_{}_seg_n1lr={}_n2lr={}_btch={}.npy'.format(patient, n1lr, n2lr, b)

n2 = np.load(filename, allow_pickle=True)

for slx in sl:
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title('Consensus')
    plt.imshow(con[slx, :, :], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title('Segmentation')
    plt.imshow(n2[slx, :, :], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle('patient = {}, slice = {}, n1={}, n2={}, b={}'.format(patient, slx, n1lr, n2lr, b))
    plt.savefig('{}_{}_{}.png'.format(patient, slx, n1lr), dpi=100)

'''
# find max sum slice
sum_idx = 0
sum_winner = 0
for slx in np.arange(con.shape[0]):
    sl_sum = con[slx, :, :].sum()
    if sl_sum > sum_winner:
        sl_sum = sum_winner
        sum_idx = slx
'''
