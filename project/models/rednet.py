#!/usr/bin/env python
# coding: utf-8

# In[3]:



from __future__ import print_function
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose,  ReLU
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np



def REDNet(n_layers = 15, n_skip=2, input_shape=(320, 320, 1), n_kernels = 64, tensor=None, enable_skip=True, enable_patches = False):
    

    n_conv_layers = n_layers
    n_deconv_layers = n_layers
    n_skip = n_skip

    def _conv_block(inputs, filters=n_kernels, kernel_size=(3, 3), strides=(1, 1), conv_id=1):
        x = Conv2D(filters, kernel_size, strides,
                               activation='relu',
                               padding='same',
                               name=f'encoder_conv{conv_id}')(inputs)
        return x
    

    def _deconv_block(inputs, filters=n_kernels, kernel_size=(3, 3), strides=(1, 1), deconv_id=1):
        x = Conv2DTranspose(filters, kernel_size, strides,
                               activation='relu',
                               padding='same',
                               name=f'decoder_deconv{deconv_id}')(inputs)
        return x

    # using keras .add for skip connections
    def _skip_block(input1, input2, skip_id=1):
        x = tf.keras.layers.add([input1, input2])
        return ReLU(name=f'skip{skip_id}_relu')(x)

    def _build_layer_list(model):
        model_layers = [layer for layer in model.layers]
        model_outputs = [layer.output for layer in model.layers]
        return model_layers, model_outputs

    # CREATE ENCODER MODEL
    encoder_inputs = Input(shape=input_shape, dtype='float32', name="encoder_inputs") # inputs skips to end
   
    for i in range(n_conv_layers):
        conv_idx = i + 1
        if conv_idx == 1:
            conv = _conv_block(encoder_inputs, conv_id=conv_idx)
        else:
            conv = _conv_block(conv, conv_id=conv_idx)

    encoded = conv
    encoder = Model(inputs=encoder_inputs, outputs=encoded, name='encoder')

    # Create encoder layer and output lists
    encoder_layers, encoder_outputs = _build_layer_list(encoder)

    # CREATE AUTOENCODER MODEL
    for i, skip in enumerate(reversed(encoder_outputs[:-1])):
        deconv_idx = i + 1
        deconv_filters = n_kernels
        if deconv_idx == n_deconv_layers:
            deconv_filters = 1
            
           
        if deconv_idx == 1:
           
            deconv = _deconv_block(encoded, filters=deconv_filters, deconv_id=deconv_idx)
            
        else:
            deconv = _deconv_block(deconv, filters=deconv_filters, deconv_id=deconv_idx)
           

        if enable_skip:
            if deconv_idx % n_skip == 0:
                skip_num = deconv_idx // n_skip
               
                #assert deconv.shape == skip.shape
                deconv = _skip_block(deconv, skip, skip_id=skip_num)
                

    decoded = deconv #(decoder_inputs)
    model = Model(inputs=encoder_inputs, outputs=decoded, name=f'REDNet{n_conv_layers}')

    # Create model layer and output lists
    model_layers, model_outputs = _build_layer_list(model)

    # CREATE DECODER MODEL
    encoded_input = Input(shape=encoded.shape[1:])
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Create decoder layer and output lists
    decoder_layers, decoder_outputs = _build_layer_list(decoder)
  
    model.summary()    

    return model, encoder, decoder





