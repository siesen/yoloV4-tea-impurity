import tensorflow as tf
import cv2
import numpy as np
import os

file_path='VOCdevkit/VOC2007/JPEGImages'
files=os.listdir(file_path)
file_new_path='VOCdevkit/VOC2007/JPEGImages_tea'

def body(x):
    x=tf.keras.layers.MaxPool2D((2,2))(x)
    x=tf.keras.layers.AvgPool2D((2,2))(x)
    # x=tf.keras.layers.MaxPool2D((2,2))(x)
    x=tf.keras.layers.MaxPool2D((2,2))(x)
    return x

for each in files:
    img=cv2.imread(os.path.join(file_path,each))
    #原图2280*5028，取2280*4864
    img_cut=img[:,82:4946]
    img_cut=np.expand_dims(img_cut,axis=0)
    img_cut=np.array(img_cut,dtype=np.float32)

    # 把2280*4864转成285*608
    model_output=body(img_cut)
    xx=model_output.numpy()[0]

    #保存下来
    cv2.imwrite(os.path.join(file_new_path,each),xx)
    # cv2.imshow('show',xx)
