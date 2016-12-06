################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import numpy as np
import os
from pylab import *
import matplotlib.pyplot as plt
import csv
import cv2
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import pickle
import random as pyrand
import sys


import tensorflow as tf

################################################################################
#Read Image
def prepareImg(img):
    img = np.array(img).astype(float32)
    img = img-mean(img)
    return img

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy", encoding='latin1').item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())




NUM_IMAGES = 50
x = tf.placeholder("float", shape=[NUM_IMAGES, 227,227,3])
y = tf.placeholder("float", shape=[NUM_IMAGES, 1])

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.constant(net_data["conv1"][0])
conv1b = tf.constant(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.constant(net_data["conv2"][0])
conv2b = tf.constant(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.constant(net_data["conv3"][0])
conv3b = tf.constant(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.constant(net_data["conv4"][0])
conv4b = tf.constant(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.constant(net_data["conv5"][0])
conv5b = tf.constant(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.constant(net_data["fc6"][0])
fc6b = tf.constant(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [NUM_IMAGES, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.constant(net_data["fc7"][0])
fc7b = tf.constant(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

# fc8
# fc(1000, relu=False, name='fc8')
# fc8W = tf.Variable(net_data["fc8"][0])
# fc8b = tf.Variable(net_data["fc8"][1])
# fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

my_fc8W = weight_variable([4096, 1])
my_fc8b = bias_variable([1])
my_fc8 = tf.nn.xw_plus_b(fc7, my_fc8W, my_fc8b)

#train model
loss = tf.reduce_mean(tf.square(y-my_fc8))
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

#############################################################################################

np.random.seed(seed=1)
trainingData = sys.argv[1]
allData = []
with open(trainingData, 'r') as mturkresultsf:
    mturkcsvreader = csv.reader(mturkresultsf)
    for row in mturkcsvreader:
        impath = row[0]
        im = prepareImg(cv2.imread(impath))
        score = float(row[2])

        allData.append(np.array([im,score,impath]))
allData = np.array(allData)
np.random.shuffle(allData)
trainSize = int(allData.shape[0] * 0.8)
trainingData = allData[:trainSize]
testData = allData[trainSize:]

def getBatch(data, size):
    indexs = np.random.choice(np.arange(data.shape[0]),size)
    samples = data[indexs,:]
    Xs = np.zeros((size, 227, 227, 3))
    Ys = np.zeros((size, 1))
    for i, sample in enumerate(samples):
        Xs[i, :, :, :] = sample[0]
        Ys[i, 0] = sample[1]
    return Xs, Ys
def getTrainingBatch(size):
    return getBatch(trainingData, size)
def getTestData(size):
    return getBatch(testData, size)

saver = tf.train.Saver()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# for i in range(1501):
#     batch = getTrainingBatch(NUM_IMAGES)
#     if i%100 == 0:
#         print("step %d, training accuracy %g"%(i,sess.run(loss, feed_dict={x:batch[0], y: batch[1]})))
#         test_xs, test_ys = getTestData(NUM_IMAGES)
#         print("test accuracy %g" % sess.run(loss, feed_dict={x: test_xs, y: test_ys}))
#         save_path = saver.save(sess, "./saves/model_%06d.ckpt"%i)
#     sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})



saver.restore(sess,"./saves/model_001500.ckpt")
test_xs, test_ys = getTestData(NUM_IMAGES)
print("test accuracy %g"%sess.run(loss, feed_dict={x: test_xs, y: test_ys}))

#sort the data and save out the best in order
i = 0
allDataScores = []
while i<len(testData):
    Xs = np.zeros((NUM_IMAGES, 227, 227, 3))
    for j in range(0,i+NUM_IMAGES):
        if i+j >= len(testData):
            break
        Xs[j, :, :, :] = testData[i+j,0]

    scores = sess.run(my_fc8, feed_dict={x: Xs})
    for j, score in enumerate(scores):
        if i + j < len(testData):
            allDataScores.append([testData[i+j,0],testData[i+j,1],score[0]])
    i+=NUM_IMAGES

allDataScores = sorted(allDataScores, key=lambda e:e[2])
for i, data in enumerate(allDataScores):
    cv2.imwrite("./imgs/%02d_%0.2f_%0.2f.jpg"%(i,data[1],data[2]),data[0])