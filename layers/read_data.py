import caffe
import scipy.io as scio
import os.path as osp
import h5py
import numpy as np
import random
import read_binaryproto
#import read_lmdb
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import global_var as GV
class input_layer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.train_batches = params['train_batches']
        self.test_batches = params['test_batches']
        self.train_data_name = params['train_data_name']
        self.test_data_name = params['test_data_name']
        self.crop_size_x = params['crop_size_x']
        self.crop_size_y = params['crop_size_y']
        self.train_pack_nums = params['train_pack_nums']
        self.test_pack_nums = params['test_pack_nums']

        this_dir = osp.dirname(__file__)
        self.data_path = osp.join(this_dir, '..',self.data_dir)

        
    def reshape(self, bottom, top):
        if self.split == 'train':
            pack_index = np.random.random_integers(1, self.train_pack_nums)
#            pack_index = 45
            h5file = h5py.File(self.data_path + self.train_data_name + str(pack_index) + '.h5', 'r')
            self.read_data = h5file['train_data'][...]
            self.read_labels = h5file['train_labels'][...]
            self.data_cases, self.data_channels, self.data_height, self.data_width = self.read_data.shape 
            rand_index = np.random.random_integers(0, self.data_cases - 11, size = self.train_batches)
            rand_index.sort()
            flage = False;
            for j in range(10):
                for i in rand_index:
                    data = self.read_data[i+j, :, :, :]
                    data = data.reshape(1,1,1,8)
                    labels = np.zeros([1])
                    labels[0] = self.read_labels[i+j,0]
                    if flage == False:
                       self.data = data
                       self.labels=labels
                       flage = True;
                    else:
                       self.data = np.concatenate((self.data,data),axis = 0)
                       self.labels = np.concatenate((self.labels,labels),axis = 0)
            cm = np.ones(self.train_batches*10)
            cm[0:self.train_batches] = 0
            self.cm = cm

            
        elif self.split == 'test':
            pack_index = np.random.random_integers(1, self.test_pack_nums)
            h5file = h5py.File(self.data_path + self.test_data_name + str(pack_index) + '.h5', 'r')
            self.read_data = h5file['test_data'][...]
            self.read_labels = h5file['test_labels'][...]
            self.data_cases, self.data_channels, self.data_height, self.data_width = self.read_data.shape 
            rand_index = np.random.random_integers(0, self.data_cases - 11, size = self.test_batches)
            rand_index.sort()
            flage = False;
            for j in range(10):
                for i in rand_index:
                    data = self.read_data[i+j, :, :, :]
                    data = data.reshape(1,1,1,8)
                    labels = np.zeros([1])
                    labels[0] = self.read_labels[i+j,0]
                    if flage == False:
                       self.data = data
                       self.labels=labels
                       flage = True;
                    else:
                       self.data = np.concatenate((self.data,data),axis = 0)
                       self.labels = np.concatenate((self.labels,labels),axis = 0)
            cm = np.ones(self.train_batches*10)
            cm[0:self.train_batches] = 0
            self.cm = cm

        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.labels.shape)
        top[2].reshape(*self.cm.shape)
        
        
    def forward(self, bottom, top):
        top[0].data[...] = self.data 
        top[1].data[...] = self.labels 
        top[2].data[...] = self.cm
        
    def backward(self, bottom, top):
        pass
    
  