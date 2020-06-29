#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
import sys  
sys.path.insert(1, '../noise_generator')
from noise_generator import *
from keras import backend as K
import math




def plot_history(history, name):

    plt.figure(figsize=(8, 5))
    plt.plot(history['PSNR'])
    plt.plot(history['val_PSNR'])
    plt.title(name)
    plt.ylabel('PSNR')
    plt.xlabel('epoch')
    plt.legend(['train,  max = '+ str(round(history['PSNR'].max(),4)) , 'val,  max = '+ str(round(history['val_PSNR'].max(),4))], loc='upper left')
#     plt.savefig(name + '.png')
    plt.show()
    
   
    
#returns as tensor
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    psnr = 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
    return psnr

#returns as numpyarray
def PSNR2 (y_true, y_pred):
    # K for symbolic Tensor
    MSE = np.mean( np.square(y_true - y_pred))
    if MSE == 0:
        return 100
    Imax = 1.0
    
    PSNR = 20 * math.log10(Imax / np.sqrt(MSE))
    return PSNR

def calculate_psnr_mean(images1, images2):
    psnrs = []
    for (i,j) in zip(images1, images2):
            
           
            psnrs.append(PSNR2(i,j))
    psnrs = np.array(psnrs)        
    return psnrs.mean()


def add_dim (imgs):
    return np.array(list((map(lambda i: np.expand_dims(i, axis=2), imgs))))

def add_motion_noise_to_images_and_add_dim(imgs, motion_type='mixed'):
    return np.array(list((map(lambda i: np.expand_dims(add_motion(i, motion_type), axis=2), imgs))))

def add_gaussian_noise_to_images_and_add_dim(imgs):
    return np.array(list((map(lambda i: np.expand_dims(add_gaussian(i), axis=2), imgs))))

