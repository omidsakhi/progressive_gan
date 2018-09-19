import tensorflow as tf
import ops
import utils


def dblock(name, inputs, num_filters, data_format):
    with tf.variable_scope(name):
        x = ops.conv2d('conv', inputs, num_filters[0], 3, data_format)
        x = ops.leaky_relu(x)
        x = ops.conv2d_down('conv_down', x, num_filters[1], 3, data_format)
        x = ops.leaky_relu(x)
        return x


def discriminator(x, resolution, cfg, is_training=True, scope='Discriminator'):
    assert(cfg.data_format == 'NCHW' or cfg.data_format == 'NHWC')

    def rname(resolution):
        return str(resolution) +'x' + str(resolution)
    def fmap(resolution):
        return cfg.resolution_to_filt_num[resolution]

    x_shape = utils.int_shape(x)
    assert(resolution == x_shape[1 if cfg.data_format == 'NHWC' else 3])
    assert(resolution == x_shape[2])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if resolution > cfg.starting_resolution:
          x1 = ops.downscale2d(x, cfg.data_format)
          x1 = ops.from_rgb('from_rgb_' + rname(resolution // 2), x1, fmap(resolution//2), cfg.data_format)
          x2 = ops.from_rgb('from_rgb_' + rname(resolution), x,fmap(resolution // 2), cfg.data_format)
          t = tf.get_variable(rname(resolution)+'_t', shape=[], dtype=tf.float32, collections=[tf.GraphKeys.GLOBAL_VARIABLES,"lerp"],
                              initializer=tf.zeros_initializer(), trainable=False)
          num_filters = [fmap(resolution), fmap(resolution // 2)]
          x2 = dblock(rname(resolution), x2, num_filters, cfg.data_format)
          x = ops.lerp_clip(x1, x2, t)
          resolution = resolution // 2
        else:
          x = ops.from_rgb('from_rgb_' + rname(resolution), x, fmap(resolution), cfg.data_format)
        while resolution >= 4:
            if resolution == 4:
                x = ops.minibatch_stddev_layer(x, cfg.data_format)
            num_filters = [fmap(resolution), fmap(resolution // 2)]
            x = dblock(rname(resolution), x, num_filters, cfg.data_format)
            resolution = resolution // 2        
        
        x = ops.dense('2x2', x, fmap(resolution), cfg.data_format)
        x = ops.leaky_relu(x)

        x = ops.dense('output', x, 1, cfg.data_format)

        return x


def gblock(name, inputs, filters, data_format):
    with tf.variable_scope(name):
        x = ops.conv2d_up('conv_up', inputs, filters, 3, data_format)
        x = ops.leaky_relu(x)
        x = ops.pixel_norm(x, data_format)
        x = ops.conv2d('conv', x, filters, 3, data_format)
        x = ops.leaky_relu(x)
        x = ops.pixel_norm(x, data_format)
        return x


def generator(x, last_layer_resolution, cfg, is_training=True, scope='Generator'):
    def rname(resolution):
        return str(resolution) + 'x' + str(resolution)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("4x4"):
            fn4 = cfg.resolution_to_filt_num[4]            
            x = ops.pixel_norm(x, cfg.data_format)
            x = ops.dense('dense', x, 4 * 4 * fn4, cfg.data_format)
            if cfg.data_format == 'NHWC':
                x = tf.reshape(x, [-1, 4, 4, fn4])
            else:
                x = tf.reshape(x, [-1, fn4, 4, 4])
            x = ops.leaky_relu(x)
            x = ops.pixel_norm(x, cfg.data_format)
            x = ops.conv2d('conv', x, fn4, 3, cfg.data_format)
            x = ops.leaky_relu(x)
            x = ops.pixel_norm(x, cfg.data_format)            
        resolution = 8
        prev_x = None
        while resolution <= last_layer_resolution:            
            filt_num = cfg.resolution_to_filt_num[resolution]
            prev_x = x
            x = gblock(rname(resolution), x, filt_num, cfg.data_format)
            resolution *= 2
        resolution = resolution // 2
        if resolution > cfg.starting_resolution:            
            t = tf.get_variable(rname(resolution) + '_t', shape=[], collections=[tf.GraphKeys.GLOBAL_VARIABLES,"lerp"],
                                dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
            x1 = ops.to_rgb('to_rgb_'+rname(resolution // 2), prev_x, cfg.data_format)
            x1 = ops.upscale2d(x1, cfg.data_format)
            x2 = ops.to_rgb('to_rgb_'+rname(resolution), x, cfg.data_format)
            x = ops.lerp_clip(x1, x2, t)
        else:
            x = ops.to_rgb('to_rgb_'+rname(resolution), x, cfg.data_format)
        x_shape = utils.int_shape(x)        
        assert(resolution == x_shape[1 if cfg.data_format == 'NHWC' else 3])
        assert(resolution == x_shape[2])        
        return x
