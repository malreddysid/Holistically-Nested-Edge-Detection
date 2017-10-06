# This file defines the model and returns the placeholders and outputs.
# The network is the similar to the one defined in the reference caffe implementation

import numpy as np
import tensorflow as tf
import model_helper as mh

def build_graph():
    input = tf.placeholder(tf.float32, [1, None, None, 3])
    label = tf.placeholder(tf.float32, [1, None, None, 1])

    input_pad = mh.pad(input=input, size=35)

    conv1_1 = mh.conv(input=input_pad, kernel_h=3, kernel_w=3, num_output=64, stride_h=1, stride_w=1, name='conv1_1')
    conv1_2 = mh.conv(input=conv1_1, kernel_h=3, kernel_w=3, num_output=64, stride_h=1, stride_w=1, name='conv1_2')
    pool1 = mh.max_pool(input=conv1_2, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, name='pool1')

    conv2_1 = mh.conv(input=pool1, kernel_h=3, kernel_w=3, num_output=128, stride_h=1, stride_w=1, name='conv2_1')
    conv2_2 = mh.conv(input=conv2_1, kernel_h=3, kernel_w=3, num_output=128, stride_h=1, stride_w=1, name='conv2_2')
    pool2 = mh.max_pool(input=conv2_2, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, name='pool2')

    conv3_1 = mh.conv(input=pool2, kernel_h=3, kernel_w=3, num_output=256, stride_h=1, stride_w=1, name='conv3_1')
    conv3_2 = mh.conv(input=conv3_1, kernel_h=3, kernel_w=3, num_output=256, stride_h=1, stride_w=1, name='conv3_2')
    conv3_3 = mh.conv(input=conv3_2, kernel_h=3, kernel_w=3, num_output=256, stride_h=1, stride_w=1, name='conv3_3')
    pool3 = mh.max_pool(input=conv3_3, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, name='pool3')

    conv4_1 = mh.conv(input=pool3, kernel_h=3, kernel_w=3, num_output=512, stride_h=1, stride_w=1, name='conv4_1')
    conv4_2 = mh.conv(input=conv4_1, kernel_h=3, kernel_w=3, num_output=512, stride_h=1, stride_w=1, name='conv4_2')
    conv4_3 = mh.conv(input=conv4_2, kernel_h=3, kernel_w=3, num_output=512, stride_h=1, stride_w=1, name='conv4_3')
    pool4 = mh.max_pool(input=conv4_3, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, name='pool4')

    conv5_1 = mh.conv(input=pool4, kernel_h=3, kernel_w=3, num_output=512, stride_h=1, stride_w=1, name='conv5_1')
    conv5_2 = mh.conv(input=conv5_1, kernel_h=3, kernel_w=3, num_output=512, stride_h=1, stride_w=1, name='conv5_2')
    conv5_3 = mh.conv(input=conv5_2, kernel_h=3, kernel_w=3, num_output=512, stride_h=1, stride_w=1, name='conv5_3')

    ### DSN conv 1 ###
    score_dsn1 = mh.conv(input=conv1_2, kernel_h=1, kernel_w=1, num_output=1, stride_h=1, stride_w=1, name='score_dsn1')
    upscore_dsn1 = mh.crop(score_dsn1, input)
    dsn1_loss = mh.sigmoid_loss(input=upscore_dsn1, labels=label, name='dsn1_loss')
    sigmoid_dsn1 = tf.nn.sigmoid(x=upscore_dsn1, name='sigmoid_dsn1')

    ### DSN conv 2 ###
    score_dsn2 = mh.conv(input=conv2_2, kernel_h=1, kernel_w=1, num_output=1, stride_h=1, stride_w=1, name='score_dsn2')
    score_dsn2_up = mh.de_conv(input=score_dsn2, up=2, kernel_h=4, kernel_w=4, num_output=1, stride_h=2, stride_w=2, name='score_dsn2_up')
    upscore_dsn2 = mh.crop(score_dsn2_up, input)
    dsn2_loss = mh.sigmoid_loss(input=upscore_dsn2, labels=label, name='dsn2_loss')
    sigmoid_dsn2 = tf.nn.sigmoid(x=upscore_dsn2, name='sigmoid_dsn2')

    ### DSN conv 3 ###
    score_dsn3 = mh.conv(input=conv3_3, kernel_h=1, kernel_w=1, num_output=1, stride_h=1, stride_w=1, name='score_dsn3')
    score_dsn3_up = mh.de_conv(input=score_dsn3, up=4, kernel_h=8, kernel_w=8, num_output=1, stride_h=4, stride_w=4, name='score_dsn3_up')
    upscore_dsn3 = mh.crop(score_dsn3_up, input)
    dsn3_loss = mh.sigmoid_loss(input=upscore_dsn3, labels=label, name='dsn3_loss')
    sigmoid_dsn3 = tf.nn.sigmoid(x=upscore_dsn3, name='sigmoid_dsn3')

    ### DSN conv 4 ###
    score_dsn4 = mh.conv(input=conv4_3, kernel_h=1, kernel_w=1, num_output=1, stride_h=1, stride_w=1, name='score_dsn4')
    score_dsn4_up = mh.de_conv(input=score_dsn4, up=8, kernel_h=16, kernel_w=16, num_output=1, stride_h=8, stride_w=8, name='score_dsn4_up')
    upscore_dsn4 = mh.crop(score_dsn4_up, input)
    dsn4_loss = mh.sigmoid_loss(input=upscore_dsn4, labels=label, name='dsn4_loss')
    sigmoid_dsn4 = tf.nn.sigmoid(x=upscore_dsn4, name='sigmoid_dsn4')

    ### DSN conv 5 ###
    score_dsn5 = mh.conv(input=conv5_3, kernel_h=1, kernel_w=1, num_output=1, stride_h=1, stride_w=1, name='score_dsn5')
    score_dsn5_up = mh.de_conv(input=score_dsn5, up=8, kernel_h=32, kernel_w=32, num_output=1, stride_h=16, stride_w=16, name='score_dsn5_up')
    upscore_dsn5 = mh.crop(score_dsn5_up, input)
    dsn5_loss = mh.sigmoid_loss(input=upscore_dsn5, labels=label, name='dsn5_loss')
    sigmoid_dsn5 = tf.nn.sigmoid(x=upscore_dsn5, name='sigmoid_dsn5')

    ### Concat and multiscale weight layer ###
    concat_upscore = tf.concat([upscore_dsn1, upscore_dsn2, upscore_dsn3, upscore_dsn4, upscore_dsn5], axis=3, name='concat_upscore')
    upscore_fuse = mh.conv(input=concat_upscore, kernel_h=1, kernel_w=1, num_output=1, stride_h=1, stride_w=1, name='upscore_fuse', num_inp=5)
    fuse_loss = mh.sigmoid_loss(input=upscore_fuse, labels=label, name='fuse_loss')
    sigmoid_fuse = tf.nn.sigmoid(x=upscore_fuse, name='sigmoid_fuse')

    total_loss = tf.add_n([dsn1_loss, dsn2_loss, dsn3_loss, dsn4_loss, dsn5_loss, fuse_loss])

    return input, label, total_loss, [sigmoid_dsn1, sigmoid_dsn2, sigmoid_dsn3, sigmoid_dsn4, sigmoid_dsn5, sigmoid_fuse]
