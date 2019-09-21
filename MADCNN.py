# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:06:16 2018

@author: yxli
"""

from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Activation, Multiply, Add, MaxPooling2D, Concatenate
from keras.models import Model
from keras import backend as K
import numpy as np
import keras
import cv2
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"		######note######

img_width = 520
img_height = 520
channels_input= 2
channels_output = 1
########################################################################################################################################################################
########################################################################################################################################################################
if K.image_data_format() == 'channels_first':
    input_shape = (channels_input*2, img_width, img_height)    
else:
    input_shape = (img_width, img_height, channels_input*2)    

def data_find_to_ave(data):
    return data[:,:,:,0:2]

def data_find_to_train(data):
    return data[:,:,:,2:4]

def image_fusing(shallow_network):
    im_1 = shallow_network[:,:,:,0:1]
    im_2 = shallow_network[:,:,:,1:2]
    im_map = shallow_network[:,:,:,2:3]
    im_map_inv = 1 - im_map
    
    return im_1 * im_map + im_2 * im_map_inv

########################################################################################################################################################################
def MFE(inpt):
    
    x_0 = Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(2, 2))(inpt)
    x_0 = Activation('relu')(x_0)
    
    x_1 = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x_1 = Activation('relu')(x_1)
    
    x = Concatenate()([x_0, x_1])
    
    return x

def Attention(input):
    model_mask = Conv2D(32, (3,3), padding='same', activation='relu', use_bias=True)(input)    
    model_mask = Conv2D(32, (3,3), padding='same', activation='sigmoid', use_bias=True)(model_mask)
    res_t = keras.layers.Multiply()([input,model_mask])
    
    return res_t

def MABlock(input):
    model = MFE(input)  
    model = Attention(model)   

    return model

def MADCNN(input_shape,channels_output):
    inputs = Input(shape=input_shape)

    temp_layer_ab = keras.layers.core.Lambda(data_find_to_ave)
    inputs_ab = temp_layer_ab(inputs)

    temp_layer_train = keras.layers.core.Lambda(data_find_to_train)
    inputs_train = temp_layer_train(inputs)    
      
    shallow_network = MABlock(inputs_train)    
    shallow_network = MABlock(shallow_network)         
    shallow_network = MABlock(shallow_network)   
    shallow_network = MABlock(shallow_network)         
    shallow_network = MABlock(shallow_network)   
    shallow_network = MABlock(shallow_network)              
    
    shallow_network = Conv2D(channels_output, (3,3), padding='same', activation='sigmoid', use_bias=True)(shallow_network)
    
    shallow_network = keras.layers.concatenate([inputs_ab, shallow_network], axis=-1)    
    
    temp_layer_image_fusing = keras.layers.core.Lambda(image_fusing)
    shallow_network = temp_layer_image_fusing(shallow_network)
   
    model = Model(inputs=inputs, outputs=shallow_network)
    return model


#########################################################################################################################################################################predict

weights = './weight_file/weights.hdf5'
testpath = './test_data/Lytro/'
resultpath = './result/'
model = MADCNN(input_shape,channels_output)		######note######
model.load_weights(weights)

dirlist_temp = os.listdir(testpath)   
dirlist = []
for file in dirlist_temp:
    dirlist.append(os.path.join(testpath,file))
dirlist.sort()

test_sample = np.zeros((520,520,4),dtype='float32')
#result = np.zeros((20,520,520,3),dtype='float32')
####################dont't touch####################
T = []
for i in range(20):
    im1 = cv2.imread(dirlist[2*i]).astype('float32')
    im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = im1 / 255.0
    im_gray1 = im_gray1/255
    
    im2 = cv2.imread(dirlist[2*i+1]).astype('float32')
    im_gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2 = im2 / 255.0
    im_gray2 = im_gray2/255
    
    b1,g1,r1 = cv2.split(im1)
    b2,g2,r2 = cv2.split(im2)
    
    test_sample[:,:,0] = b1
    test_sample[:,:,1] = b2
    test_sample[:,:,2] = im_gray1 - np.mean(im_gray1)
    test_sample[:,:,3] = im_gray2 - np.mean(im_gray2)
    testdata = np.expand_dims(test_sample,axis=0)
    t1= time.time() 
    result_b = model.predict(testdata)
    result_b = result_b[0,:,:,:]
    t2= time.time() 
    T.append(t2-t1)
    
    test_sample[:,:,0] = g1
    test_sample[:,:,1] = g2
    testdata = np.expand_dims(test_sample,axis=0)
    result_g = model.predict(testdata)
    result_g = result_g[0,:,:,:]
    
    test_sample[:,:,0] = r1
    test_sample[:,:,1] = r2
    testdata = np.expand_dims(test_sample,axis=0)
    result_r = model.predict(testdata)
    result_r = result_r[0,:,:,:]
    result = cv2.merge([result_b,result_g,result_r])
    rtpath = resultpath + str(i+1) +'.bmp'
    cv2.imwrite(rtpath,result*255)
    
