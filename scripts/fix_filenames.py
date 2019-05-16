'''
this script renames .json and .h5 files... they were named wrong due to bug
'''


import os

path = '/scratch/ij405/zhi_models'

# get files
file_list = os.listdir('../raw_data/')

arch_files = [fl for fl in file_list if fl[-5:] == '.json']
weight_files = [fl for fl in file_list if fl[-3:] == '.h5']

for arch in arch_files:
    
    start_idx = arch.find('architecture')
    params = arch[start_idx+13:-5] + '_'
    
    if arch[start_idx-2] == '_':
        newname = arch[:start_idx-1] + params + 'architecture.json'
    else:
        newname = arch[:start_idx] + params + 'architecture.json'
    os.rename(arch, newname)

for wght in weight_files:
    
    start_idx = wght.find('weights')
    params = wght[start_idx+8:-3] + '_'
    
    if wght[start_idx-2] == '_':
        newname = wght[:start_idx-1] + params + 'weights.h5'
    else:
        newname = wght[:start_idx] + params + 'weights.h5'
    
    os.rename(wght, newname)