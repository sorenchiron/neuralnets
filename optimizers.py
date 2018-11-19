# ###########  ###########
# Author: Danping LIAO
# Affiliation: Zhejiang University
# Copyright: 2018
# ###########  ###########
from __future__ import print_function
from __future__ import division
import numpy as np 

def nested_to_vector(list_of_weights_list):
    list_of_shape_list=[]
    weights_vec_list=[]
    for weights_list in list_of_weights_list:
        shape_list = []
        for weight in weights_list:
            weights_vec_list.append(weight.reshape(-1))
            shape_list.append(weight.shape)
        list_of_shape_list.append(shape_list)
    weight_vec = np.concatenate(weights_vec_list)
    return weight_vec,list_of_shape_list

def restore_nested_from_vector(weight_vec,list_of_shape_list):
    list_of_weights_list=[]
    consumed_length=0
    for shape_list in list_of_shape_list:
        weights_list=[]
        for shape in shape_list:
            this_consumed_length = np.product(shape)
            weight = weight_vec[consumed_length:consumed_length+this_consumed_length].reshape(shape)
            weights_list.append(weight)
            consumed_length+=this_consumed_length
        list_of_weights_list.append(weights_list)
    return list_of_weights_list

class Optimizer:
    def __init__(self):
        self.L1 = self.L1_regularizer
        self.L2 = self.L2_regularizer
    def L1_regularizer(self,w,regular_gamma):
        return np.ones_like(w)*regular_gamma
    def L2_regularizer(self,w,regular_gamma):
        return regular_gamma*2*w

class SGDOptimizer(Optimizer):
    def __init__(self,model,
        lr=0.1,
        decay=0.99,
        max_iters=10000,
        stop_decrement=1e-7,
        stop_lr=1e-7,
        regular_gamma=0.1,
        regularizer='L2'):
        Optimizer.__init__(self)
        iters=0
        last_loss=None
        self.__dict__.update(locals())

    def run(self,loss_val=None):
        if self.last_loss is None:
            # init the loss using a big value 
            self.last_loss = loss_val+1e3 
        self.lr *= self.decay
        w,wshapes = nested_to_vector(self.model.collect_weights())
        g,gshapes = nested_to_vector(self.model.collect_gradients())
        assert w.shape == g.shape
        regluar = self.__dict__[self.regularizer](w,self.regular_gamma) if self.regularizer else 0
        new_w = w - self.lr * (g + regluar)
        new_w_nested = restore_nested_from_vector(new_w,wshapes)
        self.model.assign_weights(new_w_nested)
        # check optimization stop criterion
        self.iters = self.iters+1
        loss_decrement = self.last_loss - loss_val
        self.last_loss = loss_val
        if self.lr<=self.stop_lr:
            print('learning rate vanished')
            return False 
        elif self.iters >= self.max_iters:
            print('Maximum iterations reached')
            return False
        elif (loss_decrement>=0 and loss_decrement<self.stop_decrement):
            print('loss is not decreasing')
            return False
        else:
            return True


class MomentumOptimizer(Optimizer):
    def __init__(self,model,
        lr=0.1,
        decay=0.99,
        max_iters=10000,
        gamma=0.9,
        stop_decrement=1e-7,
        stop_lr=1e-7,
        regular_gamma=0.1,
        regularizer='L2'):
        Optimizer.__init__(self)
        iters=0
        last_loss=None
        m=None
        self.__dict__.update(locals())

    def run(self,loss_val=None):
            if self.last_loss is None:
                # init the loss using a big value 
                self.last_loss = loss_val+1e3 
            self.lr *= self.decay
            w,wshapes = nested_to_vector(self.model.collect_weights())
            g,gshapes = nested_to_vector(self.model.collect_gradients())
            assert w.shape == g.shape
            # init momentum
            if self.m is None:
                self.m = g
            # gradient taken
            g = self.gamma*self.m + (1-self.gamma)*g
            # regularizer
            regluar = self.__dict__[self.regularizer](w,self.regular_gamma) if self.regularizer else 0
            # update momentum
            self.m = g
            new_w = w - self.lr * (g+regluar)
            new_w_nested = restore_nested_from_vector(new_w,wshapes)
            self.model.assign_weights(new_w_nested)
            # check optimization stop criterion
            self.iters = self.iters+1
            loss_decrement = self.last_loss - loss_val
            self.last_loss = loss_val
            if self.lr<=self.stop_lr:
                print('learning rate vanished')
                return False 
            elif self.iters >= self.max_iters:
                print('Maximum iterations reached')
                return False
            elif (loss_decrement>=0 and loss_decrement<self.stop_decrement):
                print('loss is not decreasing')
                return False
            else:
                return True


class BatchMomentumOptimizer(Optimizer):
    def __init__(self,model,
        lr=0.1,
        decay=0.99,
        max_iters=10000,
        gamma=0.9,
        stop_decrement=1e-7,
        stop_lr=1e-7,
        regular_gamma=0.01,
        regularizer='L2'):
        Optimizer.__init__(self)
        iters=0
        last_loss=None
        m=None
        self.__dict__.update(locals())

    def run(self,batch_x,batch_y,*args_for_model_forward,**kws_for_model_forward):
            self.lr *= self.decay
            w,wshapes = nested_to_vector(self.model.collect_weights())
            gradients_along_batch=[]
            loss_along_batch=[]
            for x,y in zip(batch_x,batch_y):
                current_loss_val = self.model(x,*args_for_model_forward,label=y,**kws_for_model_forward)
                g,gshapes = nested_to_vector(self.model.collect_gradients())
                assert w.shape == g.shape
                gradients_along_batch.append(g)
                loss_along_batch.append(current_loss_val)
            loss_val = np.average(loss_along_batch)
            # immediate gradient
            g = sum(gradients_along_batch) / len(gradients_along_batch)
            # init momentum using g if necessary
            if self.m is None:
                self.m = g
            # gradient taken
            g_taken = self.gamma*self.m + (1-self.gamma)*g
            # update momentum
            self.m = g_taken
            # get regularizer
            regular = self.__dict__[self.regularizer](w,self.regular_gamma) if self.regularizer else 0
            new_w = w - self.lr * (g_taken + regular) 
            # restore weight shapes from vector.
            new_w_nested = restore_nested_from_vector(new_w,wshapes)
            self.model.assign_weights(new_w_nested)
            # init last_loss
            if self.last_loss is None:
                # init the loss using a big value 
                self.last_loss = loss_val+1e3 
            # check optimization stop criterion
            self.iters = self.iters+1
            loss_decrement = self.last_loss - loss_val
            self.last_loss = loss_val
            if self.lr<=self.stop_lr:
                print('learning rate vanished')
                return False 
            elif self.iters >= self.max_iters:
                print('Maximum iterations reached')
                return False
            elif (loss_decrement>=0 and loss_decrement<self.stop_decrement):
                print('loss is not decreasing')
                return False
            else:
                return True