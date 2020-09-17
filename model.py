from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import re
import scipy
import shutil, sys  

from collections import OrderedDict
import imageio
import tensorflow as tf
import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import threading
import time
from sklearn import metrics
#import utils
global n_classes
#import triplet_loss as tri
import os.path
#from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
n_classes = 2
lr = tf.placeholder(tf.float32)      # Learning rate to be fed


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################


def activation(x):
    return tf.nn.swish(x)
    
def conv2d(name, l_input, w, b, s, p):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = l_input+b

    return l_input

def batchnorm(conv, isTraining, name='bn'):
    return tf.layers.batch_normalization(conv, training=isTraining, name="bn"+name)

def initializer(in_filters, out_filters, name, k_size=3):
    w1 = tf.get_variable(name+"W", [k_size, k_size, in_filters, out_filters], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name+"B", [out_filters], initializer=tf.truncated_normal_initializer())
    return w1, b1


def residual_block(in_x, in_filters, out_filters, stride, isDownSampled, name, isTraining, k_size=3):
    global ema_gp
    # first convolution layer
    if isDownSampled:
      in_x = tf.nn.avg_pool(in_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
      
    x = batchnorm(in_x, isTraining, name=name+'FirstBn')
    x = activation(x)
    w1, b1 = initializer(in_filters, in_filters, name+"first_res", k_size=k_size)
    x = conv2d(name+'r1', x, w1, b1, 1, "SAME")

    # second convolution layer
    x = batchnorm(x, isTraining, name=name+'SecondBn')
    x = activation(x)
    w2, b2 = initializer(in_filters, out_filters, name+"Second_res",k_size=k_size)
    x = conv2d(name+'r2', x, w2, b2, 1, "SAME")
    
    if in_filters != out_filters:
        difference = out_filters - in_filters
        left_pad = difference // 2
        right_pad = difference - left_pad
        identity = tf.pad(in_x, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
        return x + identity
    else:
        return in_x + x




def ResNet(_X, isTraining):
    global n_classes
    w1 = tf.get_variable("initWeight", [7, 7, 3, 96], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("initBias", [96], initializer=tf.truncated_normal_initializer())
    initx = conv2d('conv1', _X, w1, b1, 4, "VALID")
    
    filters_num = [96,128,256]
    block_num = [2,4,3]
    l_cnt = 1
    x = initx
    
    # ============Feature extraction network with kernel size 3x3============
    
    for i in range(len(filters_num)):
        for j in range(block_num[i]):
          
            if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
                x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'ResidualBlock%d'%(l_cnt), isTraining)
                print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            else:
                x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'ResidualBlock%d'%(l_cnt), isTraining)
                print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
            l_cnt +=1
            print("first,x")
            print(x.get_shape().as_list())
    
    layer_33 = x
    x = initx
    
    # ============Feature extraction network with kernel size 5x5============
    for i in range(len(filters_num)):
        for j in range(block_num[i]):
          
            if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
                x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'Residual5Block%d'%(l_cnt), isTraining, k_size=5)
                print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            else:
                x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'Residual5Block%d'%(l_cnt), isTraining, k_size=5)
                print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
            l_cnt +=1
    layer_55 = x
    print("Layer33's shape", layer_33.get_shape().as_list())
    print("Layer55's shape", layer_55.get_shape().as_list())

    x = tf.concat([layer_33, layer_55], 3) #3 ,means cocat in chanles
    print(x.get_shape().as_list())
    #x shape=(128,3,3,256)
    # ============ Classifier Learning============
    
    x_shape = x.get_shape().as_list()
    dense1 = x_shape[1]*x_shape[2]*x_shape[3]  #3*3*256
    W = tf.get_variable("featW", [dense1, 256], initializer=tf.truncated_normal_initializer()) #shape of (3*3*256, 128)
    b = tf.get_variable("featB", [256], initializer=tf.truncated_normal_initializer())  
    dense1 = tf.reshape(x, [-1, dense1])   #(128, 256*3*3) kol sora  1,256*3*3  , then 128 pic 
    print("dense")
    print(dense1.get_shape().as_list())

    feat = tf.nn.softmax(tf.matmul(dense1, W) + b)  #return 128*128   every pic has 128 feature (because 128 filter)
    print("feat")
    print(feat.get_shape().as_list())

    
    with tf.variable_scope('Final'):


        x = batchnorm(x, isTraining, name='FinalBn')
        x = activation(x)
        wo, bo=initializer(filters_num[-1]*2, n_classes, "FinalOutput") # shape of 3*3 with 128 chanles and 3adad7om=2
        print("wo")
        print(wo.get_shape().as_list())
        print(x.get_shape().as_list())
        x = conv2d('final', x, wo, bo, 1, "SAME") #x shape of 3*3 with 2 chanles w 3adad7om 128  
        print(x.get_shape().as_list())

        saliency = tf.argmax(x, 3)
        print("saly")
        print(saliency.get_shape().as_list())

        x=tf.reduce_mean(x, [1, 2])
        print(x.get_shape().as_list())
        W = tf.get_variable("FinalW", [n_classes, n_classes], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable("FinalB", [n_classes], initializer=tf.truncated_normal_initializer())

        out = tf.matmul(x, W) + b
        print("out")
        print(out.get_shape().as_list()) #128  * 2 

                            
                    

    return out, feat, saliency




#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list(image_list_file, training_img_dir):

    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        #print(line)
        filename, label = line[:-1].split(' ')
        #filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
        
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=64):
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn



def setup_inputs(sess, filenames, training_img_dir, image_size=64, crop_size=64, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')
    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)

def softmax(y):
  res = []
  for i in y:
    res.append(np.exp(i) / np.sum(np.exp(i), axis=0))
  return res

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def bulid_model():
    img_path = './HTJ8RLXVHP.jpg' #path of any image
    text_file = open("train.txt", "w")
    text_file.write('%s 0\n'%(img_path))
    text_file.close()
    ####### Build Graph ########
    n_classes = 2
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    train_data, train_labels, filelist1, glen1 = setup_inputs(sess, 'train.txt', img_path, batch_size=1)
    with tf.variable_scope("ResNet", reuse=tf.AUTO_REUSE) as scope:
        pred, feat, _ = ResNet(train_data, False)

    ##################################################################################

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def file_config(img_path):
    text_file = open("test.txt", "w")
    text_file.write('%s 0\n'%(img_path))
    text_file.close()



def detect(img_path):
    file_config(img_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print(img_path)
    ###### Detect ######
    tf.global_variables_initializer()
    saver = tf.train.Saver()
    ckpt_dir = './checkpoints/tf_deepUD_tri_model_all.ckpt'
    saver.restore(sess, ckpt_dir)
    pth = 'test.txt'
    test_data, test_labels, filelist2test, tlen1test = setup_inputs(sess,pth, img_path, batch_size=1,isTest=True)

    with tf.variable_scope("ResNet",reuse=tf.AUTO_REUSE) as scope:
        testpred, _, saliencyT = ResNet(test_data, False)
    result=sess.run(testpred)
    result = np.array(result, dtype=np.longdouble)
    percentage = softmax(result)
    fake_percentage = percentage[0][0]*100
    real_percentage = percentage[0][1]*100
    print(img_path)
    return fake_percentage, real_percentage