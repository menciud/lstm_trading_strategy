import _init_paths
import caffe
import tools
import os.path as osp
import numpy as np
from caffe import layers as L, params as P, to_proto
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mping

import os.path as osp
import scipy.io as scio
this_dir = osp.dirname(__file__)
  
def simpleNN(split):
    data, labels ,clip_markers = L.Python(module = 'read_data', 
                                   layer = 'input_layer',
                                   ntop = 3,
                                   param_str = str(dict(split=split, 
                                                        data_dir = 'data/',
                                                        train_data_name = 'train_data_patch',
                                                        test_data_name = 'test_data_patch',
                                                        train_batches = 25,
                                                        test_batches = 25,
                                                        crop_size_x = 64,
                                                        crop_size_y = 64,
                                                        train_pack_nums = 1,
                                                        test_pack_nums = 1
                                                        )))
    result = L.InnerProduct(data, num_output = 32, 
                            weight_filler = dict(type = 'xavier'),
                            bias_filler = dict(type = 'constant'))
    result = L.ReLU(result, in_place=True)                                                                             
    result = L.InnerProduct(result, num_output = 32, 
                            weight_filler = dict(type = 'xavier'),
                            bias_filler = dict(type = 'constant'))
    result = L.ReLU(result, in_place=True)
    if split == 'train':  
                              
        result = L.Reshape(result, reshape_param = {'shape':{'dim':[10,25,32]}})
    else:                                                    
        result = L.Reshape(result, reshape_param = {'shape':{'dim':[10,25,32]}})
    if split == 'train':  
                              
        labels = L.Reshape(labels,  reshape_param = {'shape':{'dim':[10,25]}})
    else:                                                    
        labels = L.Reshape(labels,  reshape_param = {'shape':{'dim':[10,25]}})    
    if split == 'train':  
                              
        clip_markers = L.Reshape(clip_markers, reshape_param = {'shape':{'dim':[10,25]}})
    else:                                                    
        clip_markers = L.Reshape(clip_markers,  reshape_param = {'shape':{'dim':[10,25]}})
    result =   L.LSTM(result,clip_markers, recurrent_param = { 'num_output':  32,
                                                               'weight_filler': {'type': 'uniform','min':-0.01,'max': 0.01},
                                                               'bias_filler'  : {'type' : 'constant','value' : 0}})                                                         
    result = L.InnerProduct(result, num_output = 1, 
                            weight_filler = dict(type = 'xavier'),
                            bias_filler = dict(type = 'constant'),
                              inner_product_param = {'axis' :2})                                                                       
    loss = L.Python(result, labels, module = 'pyloss', layer = 'EuclideanLossLayer', param_str = str(dict()),loss_weight=1)
    return to_proto(loss)
  
def make_net():
    with open(str(train_prototxt_dir), 'w') as f:
        f.write(str(simpleNN('train')))
    with open(str(test_prototxt_dir), 'w') as f:
        f.write(str(simpleNN('test')))

train_prototxt_dir = this_dir + '/model/train.prototxt'
test_prototxt_dir = this_dir + '/model/test.prototxt'
solver_prototxt_dir = this_dir + '/model/lstm_solver.prototxt'

if __name__=='__main__':
    make_net()
    caffe.set_device(1)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(str(solver_prototxt_dir))
#    snapshot_name = 'lstm_09_W1e-10_iter_100000'
#    state = this_dir + '/model/snapshot/' + str(snapshot_name) + '.solverstate'
#    solver.restore(str(state))
    for i in range(1000):
        solver.step(100)
    
    
    
    
    