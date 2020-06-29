
# coding: utf-8

# In[1]:
# from keras.models import Model
# from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU)
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, SpatialDropout2D, ReLU, LeakyReLU, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def Autoencoder():
    input_shape = (320,320,1)
    # Input
    i = Input(name='inputs', shape=input_shape, dtype='float32')
   
  # Encoder
    enc = Conv2D(64, (3, 3), padding='same', strides=(1,1),  activation='relu', name='encoder_conv1')(i)
    enc = MaxPooling2D((2, 2),padding= 'same', strides=(2,2) , name='encoder_pool1')(enc)
    enc = Conv2D(32, (3, 3), padding='same', strides=(1,1),  activation='relu', name='encoder_conv2')(enc)
    enc = MaxPooling2D((2, 2), padding='same', strides=(2,2), name='encoder_pool2')(enc)
    
  # Decoder
    dec = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation='relu', name='decoder_conv1')(enc)
    dec = UpSampling2D((2, 2), name='decoder_upsamp1')(dec)
    dec = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation='relu', name='decoder_conv2')(dec)
    dec = UpSampling2D((2, 2), name='decoder_upsamp2')(dec)
    out = Conv2D(1, (3, 3), padding='same', strides=(1,1), activation='sigmoid', name='outputs')(dec)
    
    model = Model(inputs=i, outputs=out)

 
    
    model.summary()
   
    
  

    return model




