# This file implements the conv, max_pool, deconv, crop, and sigmoid_loss layers to be used
# while buildinig the graph.

import numpy as np
import tensorflow as tf


def pad(input, size):
    paddings = [[0, 0], [size, size], [size, size], [0, 0]]
    output = tf.pad(input, paddings)
    return output


def conv(input, kernel_h, kernel_w, num_output, stride_h, stride_w, name,
         relu=True, padding="SAME", biased=True, num_inp=None):
    # This is used because the inputs of the fused conv layer don't have a defined
    # shape because of the deconv operator. So I hardcode the num_input to 5
    if(num_inp == None):
        num_input = input.get_shape()[-1]
    else:
        num_input = num_inp
    kernel_size = [kernel_h, kernel_w, num_input, num_output]
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('weights', shape=kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        output = tf.nn.conv2d(input, kernel, [1, stride_h, stride_w, 1], padding=padding)

        if biased:
            biases = tf.get_variable('biases', [num_output])
            output = tf.nn.bias_add(output, biases)

    if relu:
        output = tf.nn.relu(output, name=name)

    return output


def max_pool(input, kernel_h, kernel_w, stride_h, stride_w, name, padding="SAME"):
    output =  tf.nn.max_pool(input,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding,
                          name=name)
    return output;


def de_conv(input, up, kernel_h, kernel_w, num_output, stride_h, stride_w, name):
    return tf.layers.conv2d_transpose(input, num_output, (kernel_h, kernel_w), strides=(stride_h, stride_w), name=name, kernel_initializer=tf.contrib.layers.xavier_initializer())


def crop(input_1, input_2):
    shape_1 = tf.shape(input_1)
    shape_2 = tf.shape(input_2)
    # offsets for the top left corner of the crop
    offsets = [0, (shape_1[1] - shape_2[1]) // 2, (shape_1[2] - shape_2[2]) // 2, 0]
    size = [-1, shape_2[1], shape_2[2], -1]
    output = tf.slice(input_1, offsets, size)
    return output


def sigmoid_loss(input, labels, name):
    output = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=input, name=name)
    return tf.reduce_mean(tf.reduce_sum(output, reduction_indices=[1]))
