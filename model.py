# ###########  ###########
# Author: Danping LIAO
# Affiliation: Zhejiang University
# Copyright: 2018
# ###########  ###########
from __future__ import print_function
from __future__ import division
import numpy as np

class Model:
    def __init__(self):
        self.layers=[]
        self.w=None
        self.g=None
    
    def collect_weights(self,verbose=False):
        '''iterate from fist to last layer'''
        self.w = [layer.weights() for layer in self.layers]
        if verbose:
            for w,l in zip(self.w,self.layers):
                print('weights in layer:',l)
                print(w)
        return self.w
    
    def collect_gradients(self,verbose=False):
        '''iterate from last to first layer.
        result will be reversed when returned.'''
        self.g=[]
        layer_num = len(self.layers)
        L_partial_out = np.array([1])
        for i in range(layer_num):
            current_layer = self.layers[layer_num-i-1]
            gradients_against_weights = current_layer.gradients(L_partial_out)
            self.g.append(gradients_against_weights)
            if verbose:
                print('------------------layer start-------------------')
                print('layer',current_layer,'inshape',current_layer.input_shape,'outshape:',current_layer.output_shape)
                print('\t\t received g:',L_partial_out.shape,'while in-shape:',current_layer.input_shape)
                print('\t\t received L_partial_out:')
                print(L_partial_out)
            L_partial_out = current_layer.gradient_against_input(L_partial_out)
            if verbose:
                print('\t\t generated L_partial_out:')
                print(L_partial_out)
                print('\t\t generated gradients:')
                print(gradients_against_weights)
                print('-----------------layer end---------------------')
        self.g.reverse()
        return self.g

    def assign_weights(self,new_w):
        '''update weights for all layers in this model.
        arguement:
        new_w: List[ list[w1,b1], list[w2,b2]... ].
        '''
        if len(new_w) == len(self.layers):
            for layer,weights in zip(self.layers,new_w): # List[l1,l2,l3]  List[[w,b],[],[],[]]
                layer.assign_weights(weights)
            self.w=new_w
            return True
        else:
            print('Model.assign_weights:Inconsistent number of weights')
            return False

    def add(self,layer):
        self.layers.append(layer)

    def __call__(self,x,extra_outputs=[],verbose=False,**keywords):
        extra_out={} # layer -> value
        for layer in self.layers:
            x = layer.forward(x,**keywords)
            if verbose:
                print('layer:',layer,'forwards:')
                print(x)
            if layer in extra_outputs:
                extra_out[layer]=x
        extra_out_list=[extra_out[i] for i in extra_outputs]
        if len(extra_outputs)==0:
            return x
        else:
            return x,extra_out_list

    def save(self,filename):
        ''' .npy suffix will be automatically appended.'''
        if not self.w:
            self.collect_weights()
        print('saving to',filename,'with overwriting...')
        np.save(filename,self.w)
        print('model parameters saved.')

    def load(self,filename):
        '''.npy suffix will be automatically appended.'''
        filename+='.npy'
        w=None
        try:
            w = np.load(filename)
            self.assign_weights(w)
        except Exception as e:
            print('no previous checkpoints found.',e)

    def show(self):
        out_shapes = [l.output_shape for l in self.layers]
        all_shapes = out_shapes
        layer_names = [str(l.__class__) for l in self.layers]
        max_name_len = max([len(n) for n in layer_names])
        print_pattern = '|%{}s | %s'.format(max_name_len)
        structure_string = '\n'.join( \
                    [ print_pattern%(layer_name,str(shape)) for layer_name,shape in \
                        zip(layer_names,all_shapes)])
        print('==========================================================')
        print(print_pattern%('Layer_name','Output shape'))
        print('|---------------------------------------------------------')
        print(structure_string)
        print('==========================================================')
        