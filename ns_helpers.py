# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:29:57 2016

@author: tjw
"""
import theano
import numpy as np
import skimage.transform
from skimage.color import gray2rgb, rgb2gray
from lasagne.utils import floatX
from skimage.io import imread

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
BGR = False
def prep_image(im, IMAGE_W, IMAGE_H, BGR=BGR, bw=False):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h*IMAGE_W < w*IMAGE_H:
        im = skimage.transform.resize(im, (IMAGE_H, w*IMAGE_H//h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W//w, IMAGE_W), preserve_range=True)            

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_H//2:h//2+IMAGE_H-IMAGE_H//2, w//2-IMAGE_W//2: w//2-IMAGE_W//2 +IMAGE_W]        
    rawim = im.astype('uint8')
    # Shuffle axes to c01
    if bw:
        if BGR:
            im = im[:,:,::-1]
        bwim = gray2rgb(rgb2gray(im))
        im = (bwim*bw+im.astype("float64")*(1.-bw))
        if BGR:
            im = im[:,:,::-1]
    
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    if not BGR:
        im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

def deprocess(x, BGR=BGR):
    x = np.copy(x[0])
    x += MEAN_VALUES
    if not BGR:
        x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)    
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    

def get_img(i):
    return imread(i) if isinstance(i, str) else i   
    
shared_mem = {}
func_mem = {}
outputs_mem = {}

def Func(i, In, Out, updates=None):
    if i not in func_mem:
        func_mem[i] = theano.function(In, Out, updates=updates)
    return func_mem[i]

def Eval(i, Out):
    return Func(i, [], Out)()

def Shared(i, v):
    if i in shared_mem:        
        shared_mem[i].set_value(v)
    else:
        shared_mem[i]=theano.shared(v)
    return shared_mem[i]