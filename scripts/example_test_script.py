import model_lib as ml
import data_handler as dh

#%% load data

# get list of available directories
dir_list = os.listdir('../raw_data/')
dir_list = [di for di in dir_list if di[0] == '0']

# form database
print('loading data')
df = dh.create_df(dir_list, modal='flair')
print('data loaded')

# choose patient
patient = dir_list[0]

# load patients data

#%% load network


#%% evaluate patient
# for each pixel, create a patch and feed into network

# if positive, store 1, else store 0
