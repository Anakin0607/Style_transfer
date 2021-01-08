import tensorflow as tf

import numpy as np
import time
import functools
import cv2 as cv
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL.Image

def Load_img(img_path):# use the tensorflow io to read the data
    max_dim = 512 
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img,channels = 3)
    img = tf.image.convert_image_dtype(img,tf.float32)

    shape = tf.cast(tf.shape(img)[:-1],tf.float32)
    long_dim = max(shape)
    scale = max_dim/long_dim

    new_shape = tf.cast(shape * scale,tf.int32)# convert the max size to 512 pixels

    img = tf.image.resize(img,new_shape)
    img = img[tf.newaxis,:]
    return img

def imshow(image,title = None):
    if len(image.shape)>3:
        image = tf.squeeze(image,axis=0) # remove the size 1 dimensions channel

    plt.imshow(image)
    if title:
        plt.title(title)

def Tensor2Image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor,dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    
    img = PIL.Image.fromarray(tensor)
    return cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)