# ###########  ###########
# Author: Danping LIAO
# Affiliation: Zhejiang University
# Copyright: 2018
# ###########  ###########
from __future__ import print_function
from __future__ import division
import os
from tqdm import tqdm
from layers import *
from model import * 
from optimizers import *
from utils import *
import math


import argparse
parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--dataset', dest='dataset', default='mnist', help='dataset to load. or path of imgs for testing')
parser.add_argument('--phase', dest='phase', default='evaluate', help='train | inference | evaluate against training and testing sets.')
parser.add_argument('--show', dest='show', action='store_true', help='show structure')
parser.add_argument('--epochs', dest='epochs', default=50, type=int, help='epochs')
parser.add_argument('--batch', dest='batch', default=10, type=int, help='batch_size')

args = parser.parse_args()

#hyperparams
imgsize=28

#layers
dat = Input(input_shape=(imgsize,imgsize,1))
conv = Conv2D(dat,filters=3,kernel_size=3,stride=2)
act_conv = ELU(conv)
vecs = Flatten(act_conv)
layer1 = Dense(vecs,out_units=100)
act1 = ELU(layer1)
layer2 = Dense(act1,out_units=10)
softmax = Softmax(layer2)
cross_entropy = CrossEntropy(softmax)

#link layers into model
def build_training_model():
    model = Model()
    model.add(dat)
    model.add(conv)
    model.add(act_conv)
    model.add(vecs)
    model.add(layer1)
    model.add(act1)
    model.add(layer2)
    model.add(softmax)
    model.add(cross_entropy)
    return model

def train_fully_stochastic(epochs,x,y):
    #Optimizer 
    #opt = SGDOptimizer(model,lr=0.6,decay=1-1e-5,
    #    regular_gamma=0.01,
    #    regularizer='L2')
    model = build_training_model()
    model.load('model')
    opt = MomentumOptimizer(model,lr=0.01,decay=1-1e-5,
        regular_gamma=0.01,
        max_iters=60000,
        regularizer='L2')
    dataset_len,*_ = x.shape
    iters = epochs*dataset_len
    for i in range(iters):
        current_index_of_data = i%dataset_len
        current_x = x[current_index_of_data,:,:,:]
        current_y = y[current_index_of_data,:]
        loss,[logits,] = model(current_x,label=current_y,extra_outputs=[softmax,])
        print('[%06d/%06d]loss:%04.9f'%(i,iters,loss))
        if opt.run(loss_val=loss):
            continue
        else:
            print('Optimizer asked to stop the optimization')
            break
    model.save('model')
    model.show()
    print('training done')

def train_batch(epochs,x,y,batch_size):
    print('train_batch')
    model = build_training_model()
    model.load('model')
    opt = BatchMomentumOptimizer(model,lr=0.1,decay=1-1e-5,
        gamma=0.9,
        stop_decrement=1e-11,
        regular_gamma=0.0,
        regularizer='L2')
    dataset_len,*_ = x.shape
    batches_per_epoch = dataset_len//batch_size
    iters = epochs*batches_per_epoch
    for i in range(iters):
        current_index_of_batch = i%batches_per_epoch
        current_x = x[current_index_of_batch*batch_size:current_index_of_batch*batch_size+batch_size,:,:,:]
        current_y = y[current_index_of_batch*batch_size:current_index_of_batch*batch_size+batch_size,:]
        if opt.run(batch_x=current_x,batch_y=current_y):
            loss = opt.last_loss
            print('[%06d/%06d]loss:%04.9f'%(i,iters,loss))
            if math.isnan(loss):
                np.save('x',x)
                np.save('y',y)
                input('train:Nan encountered, xy saved, press to continue')
                break
            continue
        else:
            print('Optimizer asked to stop the optimization')
            break
    model.save('model')
    model.show()
    print('training done')

def test(x):
    model = build_training_model()
    model.load('model')
    Y = np.array([0,1,2,3,4,5,6,7,8,9])
    ys = []
    for xi in tqdm(x):
        _,[logits] = model(xi,label=Y,extra_outputs=[softmax])
        yi = np.dot(binarize(logits).reshape(-1),Y)
        ys.append(yi)
    return ys

def eval(x,y):
    model = build_training_model()
    model.load('model')
    correct=0
    wrong=0
    loss=0
    num,*_ =x.shape
    for xi,yi in tqdm(zip(x,y)):
        this_loss,[logits] = model(xi,label=yi,extra_outputs=[softmax])
        loss+=this_loss
        bin_yi = binarize(logits).reshape(yi.shape)
        is_correct = abs(bin_yi-yi).sum()==0
        correct +=  is_correct
        wrong +=  not is_correct
    print('loss:%.7f,accuracy:%.3f'%(loss/num,correct/num))

if __name__ == '__main__':
    train_data,train_labels,test_data,test_labels,test_numbers = load_mnist()
    x,y,x_test,y_test=train_data,train_labels,test_data,test_labels
    
    if args.show:
        model = build_training_model()
        model.show()
        exit(0)
    
    if args.phase == 'train':
        train_batch(args.epochs,x,y,args.batch)
    elif args.phase == 'evaluate':
        print('Evaluating against TRAINING SET:')
        eval(x,y)
        print('Evaluating against TESTING SET:')
        eval(x_test,y_test)
    elif args.phase == 'inference':
        tags = [str(i) for i in range(x_test.shape[0])]
        
        if args.dataset.lower() != 'mnist':
            x_test,tags=read_imgs(args.dataset,imgsize)

        ys = test(x_test)
        verbose = x_test.shape[0]<20

        with open('test.txt','w') as f:
            for i,(tag,yi) in enumerate(zip(tags,ys)):
                append_str=''
                if args.dataset.lower() == 'mnist':
                    append_str = 'label:'+str(test_numbers[i])
                line = 'image index:%s,predict:%d %s'%(tag,yi,append_str)
                f.write(line+os.linesep)
                if verbose:
                    print(line)
        print('Done, test output is written to','test.txt')





