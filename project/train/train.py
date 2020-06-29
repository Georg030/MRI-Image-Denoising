#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, absolute_import, with_statement
import sys  

sys.path.insert(0, '../models')
sys.path.insert(1, '../utils')
sys.path.insert(2, '../noise_generator')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import skimage
from rednet import REDNet
from utils import *
from noise_generator import add_motion, add_gaussian
from autoencoder import Autoencoder
from tensorflow import image
from keras import backend as K
import pandas as pd
import argparse
from sklearn.feature_extraction import image
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,default="RED")
parser.add_argument('--dataset', type=str,default="gaussian")
parser.add_argument('--motion_type', type=str,default="mixed")
parser.add_argument('--enable_skip',type=bool,default=True) 
parser.add_argument('--batch_size', type=int,default=1)
parser.add_argument('--kernel_number', type=int,default=32)
parser.add_argument('--layer_number', type=int,default=5)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument('--epochs', type=int,default=100)
parser.add_argument('--steps', type=int,default=100)
parser.add_argument('--patience', type=int,default=3)
args = parser.parse_args()



######## Functions for Training  #########
########                         #########
def generate_gaussian(clean_data, batch_size):
    batch_counter = 0
    inputs = []
    labels = []
    
    while 1 : 
        for i in range(clean_data.shape[0]):
            batch_counter += 1
            clean_img = clean_data[i] 
            clean_img = np.expand_dims(clean_img, axis=2)
            labels.append(clean_img)
            
            noised_img = add_gaussian(clean_data[i])
            noised_img = np.expand_dims(noised_img, axis=2)
            inputs.append(noised_img)
           
            if batch_counter % batch_size == 0:
                yield  np.array(inputs), np.array(labels)
                inputs = []
                labels = []

                
def generate_motion(clean_data, batch_size, motion_type = 'mixed'):
    batch_counter = 0
    inputs = []
    labels = []

    while 1 : 
        for i in range(clean_data.shape[0]):
            batch_counter += 1
            noised_img = clean_data[i]
            noised_img = add_motion(noised_img, motion_type = motion_type) 
            clean_img = clean_data[i]
            
            noised_img = np.expand_dims(noised_img, axis=2)
            inputs.append(noised_img)
            clean_img = np.expand_dims(clean_img, axis=2)
            labels.append(clean_img)
            
            if batch_counter % batch_size == 0:
                yield  np.array(inputs),  np.array(labels)
                inputs = []
                labels = []
                             
def get_train_data(path):
    f = h5py.File(path, "r") 
    train_data = f.get("train_data").value 
    val_data = f.get("val_data").value 
    f.close()
    return train_data, val_data



######### Get Data and Generator ##########    
if args.dataset == 'motion':
    data_path ="../data/motion_data.hdf5"
    train_data, val_data = get_train_data(data_path)
    train_generator = generate_motion(train_data, args.batch_size, args.motion_type)
    val_generator = generate_motion(val_data, args.batch_size, args.motion_type )

if args.dataset == 'gaussian':
    data_path ="../data/gaussian_data.hdf5"
    train_data, val_data = get_train_data(data_path)
    train_generator = generate_gaussian(train_data, args.batch_size)
    val_generator = generate_gaussian(val_data, args.batch_size)

    
    
######### Get Model #########
if args.model == 'RED':
    model, encoder, decoder = REDNet(enable_skip=args.enable_skip, n_kernels = args.kernel_number, n_layers = args.layer_number)

if args.model == 'Autoencoder':
    model = Autoencoder()


    
######### Get Optimizer #########
optimizer = tf.keras.optimizers.Adam(args.learning_rate)
model.compile(optimizer,
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[PSNR])
   
    

######### Set Callbacks #########
callbacks = [
  tf.keras.callbacks.EarlyStopping(patience=args.patience, monitor='val_loss'),
  tf.keras.callbacks.TensorBoard(log_dir='tensorboard'),
  tf.keras.callbacks.ModelCheckpoint(
            filepath='tensorboard/model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=2)]



######### Train & Save Model #########
history = model.fit_generator((train_generator), validation_data = val_generator, steps_per_epoch = args.steps // args.batch_size, validation_steps=args.steps // args.batch_size, epochs = args.epochs, verbose=1, callbacks=callbacks)


######### Save History#########
pd.DataFrame.from_dict(model.history.history).to_csv('tensorboard/history.csv',index=False)



