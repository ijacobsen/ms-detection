import os
import pandas as pd
import yaml
from matplotlib import pyplot as plt

dir_name = 'crossval_eval'
btch_sz = 8
p = 3
epoch_sz = 300
#file_name = 'log_btch{}_epochs{}'.format(btch_sz, epoch_sz)
file_name = 'log_btch{}_p{}_epochs{}'.format(btch_sz, p, epoch_sz)


network1_df = pd.DataFrame(index=['acc', 'loss', 'val_acc', 'val_loss'])
network2_df = pd.DataFrame(index=['acc', 'loss', 'val_acc', 'val_loss'])

logfile = os.path.join(os.path.join('..', dir_name), file_name)

with open(logfile, 'r') as f:
    lines = f.read().splitlines()

counter = 0
for line in lines:
    if line[:15] == 'lv1out_network1':
        patient = line[16:]
        data = yaml.load(lines[counter + 1])
        network1_df[patient] = data.values()
        data = yaml.load(lines[counter + 5])
        network2_df[patient] = data.values()
    counter = counter + 1

patient_list = network1_df.columns
patient = patient_list[0]
pat_len = len(patient_list) - 1
pat_len = 2
#%%
#~~~~~~~~~~~~~~~~~~~ PLOT VALIDATION ACCURACY ~~~~~~~~~~~~~~~~~~~
plt.plot(network1_df[patient]['val_acc'])
plt.plot(network2_df[patient]['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend(['n1', 'n2'])
plt.title('batch size = {}, {} patients'.format(btch_sz, pat_len))
plt.savefig('b{}_p{}__val_acc_dropout.png'.format(btch_sz, pat_len), dpi=100)

#~~~~~~~~~~~~~~~~~~~ PLOT TRAINING ACCURACY ~~~~~~~~~~~~~~~~~~~
plt.clf()
plt.plot(network1_df[patient]['acc'])
plt.plot(network2_df[patient]['acc'])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend(['n1', 'n2'])
plt.title('batch size = {}, {} patients'.format(btch_sz, pat_len))
plt.savefig('b{}_p{}__train_acc_dropout.png'.format(btch_sz, pat_len), dpi=100)

#~~~~~~~~~~~~~~~~~~~ PLOT VALIDATION LOSS ~~~~~~~~~~~~~~~~~~~
plt.clf()
plt.plot(network1_df[patient]['val_loss'])
plt.plot(network2_df[patient]['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(['n1', 'n2'])
plt.title('batch size = {}, {} patients'.format(btch_sz, pat_len))
plt.savefig('b{}_p{}__val_loss_dropout.png'.format(btch_sz, pat_len), dpi=100)

#~~~~~~~~~~~~~~~~~~~ PLOT TRAINING LOSS ~~~~~~~~~~~~~~~~~~~
plt.clf()
plt.plot(network1_df[patient]['loss'])
plt.plot(network2_df[patient]['loss'])
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(['n1', 'n2'])
plt.title('batch size = {}, {} patients'.format(btch_sz, pat_len))
plt.savefig('b{}_p{}__train_loss_dropout.png'.format(btch_sz, pat_len), dpi=100)