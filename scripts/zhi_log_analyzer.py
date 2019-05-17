import os
import pandas as pd
import yaml
from matplotlib import pyplot as plt

dir_name = 'zhi_eval'
btch_sz = 128
p = 12
epoch_sz = 240
n1lr = '0.003'
n2lr = '0.003'
#file_name = 'log_btch{}_epochs{}'.format(btch_sz, epoch_sz)
file_name = 'zhi_log_btch{}_p{}_epochs{}_n1lr={}_n2lr={}'.format(btch_sz, p, 
                                                                 epoch_sz, n1lr, n2lr)


network1_df = pd.DataFrame(index=['acc', 'loss', 'val_acc', 'val_loss'])
network2_df = pd.DataFrame(index=['acc', 'loss', 'val_acc', 'val_loss'])

logfile = os.path.join(os.path.join('..', dir_name), file_name)

with open(logfile, 'r') as f:
    lines = f.read().splitlines()

counter = 0
for line in lines:
    if line[:15] == 'zhi_network1':
        patient = line[16:]
        data = yaml.load(lines[counter + 1])
        network1_df[patient] = data.values()
        data = yaml.load(lines[counter + 5])
        network2_df[patient] = data.values()
    counter = counter + 1

patient_list = network1_df.columns
patient = patient_list[0]
pat_len = len(patient_list) - 1
#%%
#~~~~~~~~~~~~~~~~~~~ PLOT VALIDATION ACCURACY ~~~~~~~~~~~~~~~~~~~
plt.clf()
plt.subplot(2, 2, 1)
plt.plot(network1_df[patient]['val_acc'])
plt.plot(network2_df[patient]['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend(['n1', 'n2'])

#~~~~~~~~~~~~~~~~~~~ PLOT TRAINING ACCURACY ~~~~~~~~~~~~~~~~~~~
plt.subplot(2, 2, 2)
plt.plot(network1_df[patient]['acc'])
plt.plot(network2_df[patient]['acc'])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend(['n1', 'n2'])

#~~~~~~~~~~~~~~~~~~~ PLOT VALIDATION LOSS ~~~~~~~~~~~~~~~~~~~
plt.subplot(2, 2, 3)
plt.plot(network1_df[patient]['val_loss'])
plt.plot(network2_df[patient]['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(['n1', 'n2'])

#~~~~~~~~~~~~~~~~~~~ PLOT TRAINING LOSS ~~~~~~~~~~~~~~~~~~~
plt.subplot(2, 2, 4)
plt.plot(network1_df[patient]['loss'])
plt.plot(network2_df[patient]['loss'])
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(['n1', 'n2'])

plt.tight_layout()
plt.suptitle('zhi_n1lr={}_n2lr={}_b{}_p{}__dropout.png'.format(n1lr, n2lr, patient, btch_sz, pat_len))
plt.savefig('zhi_n1lr={}_n2lr={}_b{}_p{}__dropout.png'.format(n1lr, n2lr, patient, btch_sz, pat_len), dpi=100)
