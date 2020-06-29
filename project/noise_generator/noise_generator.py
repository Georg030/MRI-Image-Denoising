#!/usr/bin/env python
# coding: utf-8



import skimage
from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt



def add_gaussian(img):
    #variance of gaussian distribution
    variance = np.random.uniform(0.005,0.02)
    gaussian_img = skimage.util.random_noise(img, mode="gaussian",  var = variance, clip=True )
    return gaussian_img
    


def add_motion(img, motion_type = 'mixed'):
    
    img = 255 * img 
    img = img.astype(np.float)
    original = np.copy(img)
    sobel = np.power(filters.sobel(original),1.7)
  
    if motion_type == 'random': 
        motion_img = random_overlapping(img, sobel) 
    
    if motion_type == 'symmetric':
        motion_img = symmetric_overlapping(img, sobel)
        
    if motion_type == 'mixed':
        rand_int = np.random.randint(2)
        if rand_int == 0:
            motion_img = random_overlapping(img, sobel) 
        if rand_int == 1:
            motion_img = symmetric_overlapping(img, sobel)

    return motion_img/motion_img.max()


def random_overlapping(img, sobel_img):
    original = np.copy(img)
    number_overlaps = np.random.randint(5, 80)   
    max_range = np.random.randint(45,95)
    noise_multiplier = np.random.uniform(0.7,1.5)
    for i in range(number_overlaps):
        x,y = np.random.randint(-max_range,max_range), np.random.randint(-max_range,max_range)
        number_overlaps* max_range 
        factor = (np.random.uniform(0.005,0.025) *np.power(max_range, 0.18)/(np.power(number_overlaps, 0.45))) * noise_multiplier
    
        img = overlap_image(img,sobel_img,x,y,factor,original)
    return img



def symmetric_overlapping(img, img_sobel):
    original = np.copy(img)
    number_overlaps  = np.random.randint(4,21)
    noise_multiplier = np.random.uniform(0.7,1.5)
    y_trans = 0
    for i in range(number_overlaps):
        max_y = y_trans + 15
        min_y = y_trans + 5
        y_trans = np.random.randint(min_y, max_y)
        factor = ((np.random.randint(1, 5) / 325)*noise_multiplier) * (1-0.025*i)
        x_trans = np.random.randint(1,6)
        img = overlap_image(img,img_sobel,x_trans,y_trans,factor, original)
        img = overlap_image(img,img_sobel,-x_trans,-y_trans,factor,original)
    return img


def overlap_image(base_img, sobel_img, x, y, factor, original, mul = 0.5):
    base_img = np.copy(base_img)
    
    shape = base_img.shape

  
    # sobel bild wird um x,y verschoben und gekürzt, bilder werden in in die gleiche größe gebracht (Überlappungsfläche) damit man es übereinander lappen kann, anschließend in gleichen Datentypen
    sobel_img = sobel_img[int(y>=0)*y:shape[0]+y,int(x>=0)*x:shape[1]+x]*factor
    sobel_img = np.array(sobel_img, dtype=base_img.dtype)
    
    #verringerter factor des sobels außerhalb des Original-Gehirns 
    overlap_area_original_image = original[int(y<0)*-y:shape[0]-int(y>=0)*+y,int(x<0)*-x:shape[1]-int(x>=0)*x]
    sobel_img[overlap_area_original_image < np.mean(overlap_area_original_image)] *= mul
    
    # Original bild wird mit sobel addiert in Überlappungsfläche
    base_img[int(y<0)*-y:shape[0]-int(y>=0)*+y,int(x<0)*-x:shape[1]-int(x>=0)*x] += sobel_img
    return base_img



