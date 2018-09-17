
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import utils

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def default_initializer():
    return tf.variance_scaling_initializer()

def double_zero(x):
    return tf.sin(tf.sigmoid(x)*np.pi)

def lerp(a, b, t): 
    return a + (b - a) * t

def lerp_clip(a, b, t): 
    return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    
def to_rgb(name, x, data_format):
    x = conv2d(name, x, 3, 1, data_format)    
    return x

def from_rgb(name, x, filters, data_format):
    with tf.variable_scope(name):
        x = conv2d('conv', x, filters, 1, data_format)
        x = leaky_relu(x)
    return x

def upscale2d(x, data_format, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        if data_format == 'NHWC':
            s = x.shape
            x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
            x = tf.tile(x, [1, 1, factor, 1, factor, 1])
            x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        else:
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

def downscale2d(x, data_format, factor=2):
    with tf.variable_scope('Downscale2D'):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1: return x
        if data_format == 'NHWC':
            ksize = [1, factor, factor, 1]
        else:
            ksize = [1, 1, factor, factor]
        x = tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format=data_format)
        return x

def image_norm(x, data_format, epsilon=1e-8):
    with tf.variable_scope('ImageNorm'):
        if data_format == 'NHWC':
            axis = [3]
        else:
            axis = [1]
        _min = tf.reduce_min(x, axis=axis, keepdims=True)
        _max = tf.reduce_max(x, axis=axis, keepdims=True)
        x = (x - _min) / (_max - _min + 1e-8)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = (x * 2.0) - 1.0
        return x

def pixel_norm(x, data_format, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        shape = utils.int_shape(x)
        if len(shape) == 2:
            axis = 1
        else:
            axis = 3 if data_format == 'NHWC' else 1
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + epsilon)

def apply_bias(x, data_format):
    shape = utils.int_shape(x)
    assert(len(shape)==2 or len(shape)==4)
    if len(shape) == 2:        
        channels = shape[1]
    else:        
        channels = shape[3] if data_format == 'NHWC' else shape[1]
    b = tf.get_variable('bias', shape=[channels], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        if data_format == 'NHWC':
            return x + tf.reshape(b, [1, 1, 1, -1])
        else:
            return x + tf.reshape(b, [1, -1, 1, 1])            

def dense(name, x, fmaps, data_format, gain=np.sqrt(2), use_wscale=False, has_bias=True):
    with tf.variable_scope(name):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        x = tf.matmul(x, w)     
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def get_weight(shape, gain=np.sqrt(2), use_wscale=True, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

def conv2d(name, x, fmaps, kernel, data_format, has_bias=True, gain=np.sqrt(2), use_wscale=True):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[3 if data_format == 'NHWC' else 1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def conv2d_down(name, x, fmaps, kernel, data_format, gain=np.sqrt(2), has_bias=True, use_wscale=False):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[3 if data_format == 'NHWC' else 1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1,2,2,1] if data_format=='NHWC' else [1,1,2,2], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def conv2d_up(name, x, fmaps, kernel, data_format, gain=np.sqrt(2), has_bias=True, use_wscale=True):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        c = x.shape[3 if data_format == 'NHWC' else 1].value
        w = get_weight([kernel, kernel, fmaps, c], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*c)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        if data_format == 'NHWC':
            os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps]
        else:
            os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
        x = tf.nn.conv2d_transpose(x, w, os, strides=[1,2,2,1] if data_format=='NHWC' else [1,1,2,2], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def batch_norm_relu(name, inputs, is_training, data_format, relu=True, init_zero=False, scale=True):    
        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=3 if data_format == 'NHWC' else 1,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            center=True,
            scale=scale,
            training=is_training,
            fused=True,
            gamma_initializer=gamma_initializer, 
            name = name)
        if relu:
            inputs = tf.nn.relu(inputs)
        return inputs

def minibatch_stddev_layer(x, data_format, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        if data_format == 'NHWC':
            y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
            return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.
        else:
            y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
            return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

def spectral_norm():
    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
    def fun(w):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
        u_hat = u    
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)        
        return w_norm
    return fun