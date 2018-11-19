import numpy as np 
from pprint import pprint 
from multiprocessing import Pool

class Layer:
    '''Base class for deep learning layers.
    This Layer implements identity mappings f(x)=x, with NO trainable weights.
    Basic methods are implemented like:
        gradients(self,*rubbish_vals,**rubbish_keywords)
        weights(self,*rubbish_vals,**rubbish_keywords)
    that receive arbitaray parameters and returns nothing.

    '''
    def __init__(self,previous_layer=None,input_shape=None,name=None):
        '''init checks input size in a default way.
        child class should implement input_shape and output_shape properties.'''
        if not previous_layer and not input_shape:
            raise Exception('Neither previous_layer nor input_shape was specified.')
        else: # identity size for input and output by default.
            input_shape = input_shape or previous_layer.output_shape
            output_shape = input_shape
            self.__dict__.update(locals())
    def forward(self,x,**keywords):
        '''x should be saved for gradient derivations'''
        self.x = x.reshape(self.output_shape)#.reshape((-1,1))
        return self.x
    def gradient_against_input(self,L_partial_out):
        '''returns:
        partial(Loss) / partial(a_i). a_i is the output of layer i'''
        return L_partial_out
    def gradients(self,*rubbish_vals,**rubbish_keywords):
        '''returns:
        gradients of TRAINABLE parameters in this layer'''
        return []
    def weights(self,*rubbish_vals,**rubbish_keywords):
        '''returns:
        TRAINABLE parameters in this layer.
        '''
        return []
    def assign_weights(self,weights_list,*rubbish_vals,**rubbish_keywords):
        return

class Input(Layer):
    pass

class Dense(Layer):
    def __init__(self,previous_layer=None,in_units=None,out_units=1,w_init=None):
        super(self.__class__,self).__init__(previous_layer,in_units)
        input_shape = (in_units,1) if in_units is not None else previous_layer.output_shape
        in_units = np.product(input_shape)
        w = w_init if w_init is not None else np.random.normal(0,scale=1e-2,size=(out_units,in_units))
        bias = np.zeros((out_units,1))
        output_shape=(out_units,1)
        self.__dict__.update(locals())
        self.input_shape=input_shape
        return 

    def forward(self,x,**keywords):
        x = x.reshape((-1,1))
        self.x=x
        return np.matmul(self.w,x)+self.bias

    def gradient_against_w(self,L_partial_out):
        L_partial_out = L_partial_out.reshape((1,-1))
        g = np.matmul(L_partial_out.T,self.x.T)
        return g

    def gradient_against_bias(self,L_partial_out):
        L_partial_out = L_partial_out.reshape((1,-1))
        return L_partial_out

    def gradient_against_input(self,L_partial_out):
        #print('Dense received:',L_partial_out)
        L_partial_out = L_partial_out.reshape((1,-1)) # ROW
        #print('Dense',L_partial_out,self.w)
        g = np.matmul(L_partial_out,self.w)
        return g.reshape(self.input_shape)

    def gradients(self,L_partial_out,*rubbish_vals,**rubbish_keywords):
        '''gradients of Trainable Parameters in this layer'''
        return [self.gradient_against_w(L_partial_out),
        self.gradient_against_bias(L_partial_out).T]

    def weights(self,*rubbish_vals,**rubbish_keywords):
        return [self.w,self.bias]

    def assign_weights(self,weights_list):
        assert len(weights_list) == 2
        w,bias = weights_list
        assert self.w.shape == w.shape
        assert self.bias.shape == bias.shape
        self.w,self.bias = weights_list

class Softmax(Layer):
    def __init__(self,previous_layer=None,input_shape=None):
        super(self.__class__,self).__init__(previous_layer,input_shape)
        self.input_shape = input_shape or previous_layer.output_shape
        self.output_shape = self.input_shape
        self.previous_layer = previous_layer

    def forward(self,x,**keywords):
        '''return shape can be a matrix'''
        x = x.reshape((-1,1))
        out = np.exp(x) / np.exp(x).sum()
        self.x=x
        return out 

    def gradient_against_input(self,L_partial_out):
        '''flatten the x and L_partial_out first.
        restore the shape to be same as Input_shape after derivation.'''
        L_partial_out = L_partial_out.reshape((1,-1)) # row
        input_len = np.product(self.input_shape)
        exp_x = np.exp(self.x.reshape((-1,1))) # column
        sum_exp_x = exp_x.sum()
        #sum2_exp_x = sum_exp_x**2 #+ 1e-7
        #print(L_partial_out.shape,exp_x.shape,sum_exp_x.shape,sum2_exp_x.shape)
        #print(((- exp_x * exp_x.T / sum2_exp_x)).shape)
        diagonal_part = (exp_x.T/sum_exp_x) * np.identity(input_len)
        g = np.matmul(L_partial_out,((- np.matmul(exp_x,exp_x.T) / sum_exp_x)/sum_exp_x)+diagonal_part) # row
        return g.reshape(self.input_shape)

class CrossEntropy(Layer):
    def __init__(self,previous_layer=None,input_shape=None):
        super(self.__class__,self).__init__(previous_layer,input_shape)
        self.input_shape = input_shape or previous_layer.output_shape
        self.output_shape = (1,)
        self.previous_layer = previous_layer

    def forward(self,y,label,**keywords):
        y = y.reshape((-1,1))
        label = label.reshape((-1,1))
        self.__dict__.update(locals())
        return - np.matmul(label.T,np.log(y+1e-13))

    def gradient_against_input(self,L_partial_out):
        return - L_partial_out*self.label*(1/(self.y+1e-13))

class Flatten(Layer):
    def __init__(self,previous_layer=None,input_shape=None):
        super(self.__class__,self).__init__(previous_layer,input_shape)
        input_shape = input_shape or previous_layer.output_shape
        output_shape = (np.product(input_shape),1)
        self.__dict__.update(locals())

    def forward(self,x,**keywords):
        self.x = x
        return x.reshape((-1,1))

    def gradient_against_input(self,L_partial_out):
        L_partial_out = L_partial_out.reshape((-1,1))
        return L_partial_out.reshape(self.input_shape)

class ExponentialLinearUnit(Layer):
    def __init__(self,previous_layer=None,input_shape=None,alpha=1):
        super(self.__class__,self).__init__(previous_layer,input_shape)
        input_shape = input_shape or previous_layer.output_shape
        output_shape = input_shape
        self.__dict__.update(locals())
    def forward(self,x,**keywords):
        linear_part = x*(x>0)
        exp_part = self.alpha*(np.exp(x*(x<=0))-1)
        self.__dict__.update(locals())
        return linear_part+exp_part
    def gradient_against_input(self,L_partial_out):
        L_partial_out = L_partial_out.reshape(self.input_shape)
        ones = np.ones_like(self.x)
        linear_part = (self.x>=0)*ones
        exp_part = self.alpha*np.exp(self.x*(self.x<0))
        #print(L_partial_out.shape,(linear_part+exp_part).shape)
        return L_partial_out*(linear_part+exp_part)
ELU = ExponentialLinearUnit
Elu = ExponentialLinearUnit

class Conv2D(Layer):
    def __init__(self,previous_layer=None,input_shape=None,
        filters=1,
        kernel_size=3, stride=1, padding='valid'):
        super(self.__class__,self).__init__(previous_layer,input_shape)
        input_shape = input_shape or previous_layer.output_shape
        if len(input_shape)<3:
            target_input_shape = input_shape+(1,)
        else:
            target_input_shape = input_shape

        height,width,channels = target_input_shape
        #pool=Pool(8)
        self.__dict__.update(locals())
        self.output_shape = self.compute_shapes()
        self.init_weights()

    def compute_shapes(self):
        h,w,c = self.target_input_shape
        kernel_size = self.kernel_size
        stride = self.stride
        width_steps = w//stride
        height_steps = h//stride
        left_paddings = (kernel_size-1)//2
        top_paddings = left_paddings

        right_paddings = left_paddings - (w - width_steps * stride)
        bottom_paddings = top_paddings - (h - height_steps * stride)
        output_shape = (height_steps,width_steps,self.filters)
        padded_shape = (h+top_paddings+bottom_paddings,w+left_paddings+right_paddings,c)
        self.__dict__.update(locals())
        return output_shape

    def init_weights(self):
        filters = self.filters
        kernel_size = self.kernel_size
        channels = self.channels
        biases = np.zeros(filters)
        kernels = np.random.normal(0,scale=1e-2,size=(kernel_size*kernel_size*channels,filters))
        self.__dict__.update(locals())

    def forward(self,x,**keywords):
        '''single threaded forward'''
        stride = self.stride
        kernel_size = self.kernel_size
        standard_x = x.reshape(self.target_input_shape) # add possible missing channel.(h,w)->(h,w,1)
        padded_x = self.add_padding(standard_x)
        out = np.zeros(self.output_shape)
        for h_pos in range(self.height_steps):
            for w_pos in range(self.width_steps):
                h_start = h_pos*stride
                h_end = h_start+kernel_size
                w_start = w_pos*stride
                w_end = w_start+kernel_size
                image_patch = padded_x[h_start:h_end,w_start:w_end,:]
                
                out[h_pos,w_pos,:] = \
                    np.matmul(image_patch.reshape((1,-1)),self.kernels) + self.biases
        self.__dict__.update(locals())
        return out

    def multi_forward(self,x,**keywords):
        '''multi threaded forward'''
        stride = self.stride
        kernel_size = self.kernel_size
        standard_x = x.reshape(self.target_input_shape) # add possible missing channel.(h,w)->(h,w,1)
        padded_x = self.add_padding(standard_x)
        out = np.zeros(self.output_shape)
        params=[]
        for h_pos in range(self.height_steps):
            for w_pos in range(self.width_steps):
                params.append((padded_x,h_pos,w_pos,stride,kernel_size,
                    self.kernels,self.biases))
        res = self.pool.map(self.traverse_img_forward,params)
        for h_pos,w_pos,i,pix in res:
            out[h_pos,w_pos,i]=pix
        self.__dict__.update(locals())
        return out

    def traverse_img_forward(self,param):
        padded_x,h_pos,w_pos, \
        stride,kernel_size,kernels,biases=param
        h_start = h_pos*stride
        h_end = h_start+kernel_size
        w_start = w_pos*stride
        w_end = w_start+kernel_size
        image_patch = padded_x[h_start:h_end,w_start:w_end,:]
        out=[]
        for i,ki in enumerate(kernels):
            pix = (ki * image_patch).sum()+biases[i]
            out.append((h_pos,w_pos,i,pix))
        return out

    def add_padding(self,x):
        return self.valid(x)

    def valid(self,x):
        top_paddings = self.top_paddings
        left_paddings = self.left_paddings
        h,w=self.h,self.w
        background = np.zeros(self.padded_shape)
        background[top_paddings:top_paddings+h,left_paddings:left_paddings+w,:]=x
        return background

    def mirror(self,x):
        top_paddings = self.top_paddings
        left_paddings = self.left_paddings
        h,w=self.h,self.w
        # used for mirror assigning
        h_flip = np.flip(x,axis=1)
        v_flip = np.flip(x,axis=0)
        background = np.zeros(self.padded_shape)
        # place image in the center
        background[top_paddings:top_paddings:h,left_paddings:left_paddings+w,:]=x
        # place top mirror part
        background[0:top_paddings,left_paddings:left_paddings+w,:]=v_flip[h-top]
        # unfinished

    def gradient_against_w(self,L_partial_out):
        padded_x = self.padded_x
        stride = self.stride
        kernel_size = self.kernel_size
        out_partial_ks = np.zeros_like(self.kernels[0])
        out_pixel_num = np.product(self.output_shape[:2])
        kernel_volume = kernel_size**2*self.c
        expanded_x = np.zeros((out_pixel_num,kernel_volume)) 
            # outpixels x kernel_volume
        # expand image matrix
        # can deal with strides
        for h_pos in range(self.height_steps):
            for w_pos in range(self.width_steps):
                h_start = h_pos*stride
                h_end = h_start+kernel_size
                w_start = w_pos*stride
                w_end = w_start+kernel_size
                image_patch = padded_x[h_start:h_end,w_start:w_end,:]
                expanded_x[h_pos*self.height_steps+w_pos,:]=image_patch.reshape(-1)
        L_partial_out = L_partial_out.reshape((-1,self.filters))
            # outpixels x kernel_num
        g_vecs = np.matmul(expanded_x.T,L_partial_out)
            # kernel_volume x kernel_num
        return g_vecs

    def gradient_against_bias(self,L_partial_out):
        L_partial_out = L_partial_out.reshape(self.output_shape)
        g = L_partial_out.sum(axis=(0,1)).reshape(self.biases.shape)
        return g

    def gradients(self,L_partial_out,*rubbish_vals,**rubbish_keywords):
        '''returns:
        gradients of TRAINABLE parameters in this layer'''
        return [self.gradient_against_w(L_partial_out),
                self.gradient_against_bias(L_partial_out)]

    def weights(self,*rubbish_vals,**rubbish_keywords):
        '''returns:
        TRAINABLE parameters in this layer.
        '''
        return [self.kernels,self.biases]
    def assign_weights(self,weights_list,*rubbish_vals,**rubbish_keywords):
        kernels,biases = weights_list
        assert kernels.shape == self.kernels.shape
        assert biases.shape == self.biases.shape
        self.kernels = kernels
        self.biases = biases
        return