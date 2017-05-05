import _init_paths
import caffe
import numpy as np
import scipy.io as scio
image_path = './data/'
deploy =  './model/deploy.prototxt'
caffe_model = './model/snapshot/lstm_09_W1e-10_iter_100000.caffemodel'
caffe.set_device(2)
caffe.set_mode_gpu()
net = caffe.Net(deploy,caffe_model,caffe.TEST)
train_data_file = image_path + 'data_4.mat'
data = scio.loadmat(train_data_file)
train_data = data['train_x']
train_data = train_data[:, np.newaxis, np.newaxis, :]
result_all = []
for i in xrange(len(train_data)-10):
    data = train_data[i:i+10, :, :, :]
    data = data.reshape(10,1,1,8)
    net.blobs['data'].data[...] = data
    cm = np.ones([10,1])
    cm[0]=0
    net.blobs['clip_markers'].data[...] = cm
    net.forward()
    result = net.blobs['InnerProduct3'].data[9,0,0]
    result_all.append(result)
prob_dir = './data/result/train_result_1.mat'
scio.savemat(prob_dir,{'train_result_1':result_all})

    
