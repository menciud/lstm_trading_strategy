import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os.path as osp

this_dir = osp.dirname(__file__)

print this_dir

train_data_file = 'data_2.mat'
data = scio.loadmat(train_data_file)
train_data = data['train_x']
train_data = train_data[:, np.newaxis, np.newaxis, :]
train_labels = data['train_y']
#train_labels = train_labels.reshape(train_labels.shape[0])
test_data = data['test_x']
test_data = test_data[:, np.newaxis, np.newaxis, :]
test_labels = data['test_y']
#test_labels = test_labels.reshape(test_labels.shape[0])
    
h5file = h5py.File('train_data_patch1.h5', 'w')
#h5file.create_dataset('train_data', data = train_data.transpose(0,3,1,2), dtype = np.float64)
h5file.create_dataset('train_data', data = train_data, dtype = np.float64)
h5file.create_dataset('train_labels', data = train_labels, dtype = np.float64)
h5file.close()
h5file = h5py.File('test_data_patch1.h5', 'w')
#h5file.create_dataset('test_data', data = test_data.transpose(0,3,1,2), dtype = np.float64)
h5file.create_dataset('test_data', data = test_data, dtype = np.float64)
h5file.create_dataset('test_labels', data = test_labels, dtype = np.float64)
h5file.close()