# ###########  ###########
# Author: Danping LIAO
# Affiliation: Zhejiang University
# Copyright: 2018
# ###########  ###########
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import scipy.misc as sm
from glob import glob

def read_imgs(target_dir,size):
    if os.path.exists(target_dir):
        search_pattern = os.path.join(target_dir,'*.*')
        filenames = glob(search_pattern)
        imgs = []
        for filename in filenames:
            img = sm.imread(filename)
            img = sm.imresize(img,(size,size))
            if len(img.shape)==2:
                img = img.reshape((size,size,1))
            elif img.shape[2]>3:
                grey = np.average(img[:,:,:3],axis=2)
                img = grey.reshape((size,size,1))
            imgs.append(img)
        imgmat = np.stack(imgs) / 255 - 0.5
        return imgmat,filenames
    else:
        raise Exception('No such folder',target_dir)


def binarize(np_numbers):
    '''binarize probability vector into ONE-HOT vector'''
    return (np_numbers==np_numbers.max()).astype(np.int)

def gen_data(dataset_len, shape):
    if isinstance(shape,list) or isinstance(shape,tuple):
        pass
    else:# shape is a number
        shape = (shape,)
    x = np.random.rand(dataset_len,*shape)  # sum(4*(0,1))  max 4, min 0
    maxval = np.product(shape)/2
    label = x.sum(axis=1)>maxval # true or false
    # reverse logic
    label = label.astype(np.int) # (dataset_len x 1)   01000101010
    label_neg = 1-label
    # reshape
    label_pos = label.reshape((-1,1))
    label_neg = label_neg.reshape((-1,1))
    # concatenate into (dataset_len x 2)
    y = np.concatenate((label_pos,label_neg),axis=1)
    return x,y

def load_mnist():
    train_data = np.load('dataset/mnist/train_data.npy')
    train_labels = np.load('dataset/mnist/train_labels.npy')
    test_data = np.load('dataset/mnist/test_data.npy')
    test_labels = np.load('dataset/mnist/test_labels.npy')
    train_data = train_data/255 - 0.5
    test_data = test_data/255 - 0.5
    return train_data,one_hots(train_labels),test_data,one_hots(test_labels),test_labels

def one_hots(label):
    label = label.reshape(-1)
    maxval = label.max()
    minval = label.min()
    bits = maxval-minval+1
    data_len = len(label)
    label_vecs = np.zeros((data_len,bits))
    for i,number in enumerate(label):
        label_vecs[i,number]=1
    return label_vecs