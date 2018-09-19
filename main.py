import os, time
from tqdm import trange
from os import path
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_estimator  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer  # pylint: disable=E0611
from tensorflow.python.estimator import estimator  # pylint: disable=E0611
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook
import input_pipelines, utils, models, ops

DRY_RUN = False

def lerp_update_ops(resolution, value):
    name = str(resolution) + 'x' + str(resolution)
    gt = tf.get_default_graph().get_tensor_by_name('Generator/'+name+'_t:0')
    assert(gt is not None)
    dt = tf.get_default_graph().get_tensor_by_name('Discriminator/'+name+'_t:0')
    assert(dt is not None)
    return [tf.assign(gt, value), tf.assign(dt, value)]

def model_fn(features, labels, mode, cfg):
    del labels

    resolution = features['resolution']

    if mode == 'PREDICT':
        random_noise = features['random_noise'] * cfg.temperature       
        return models.generator(random_noise, resolution, cfg, is_training=False)
    
    real_images_1 = features['real_images']
    if cfg.data_format == 'NCHW':
        real_images_1 = utils.nchw_to_nhwc(real_images_1)
        real_images_2 = tf.image.flip_left_right(real_images_1)
        real_images_1 = utils.nhwc_to_nchw(real_images_1)
        real_images_2 = utils.nhwc_to_nchw(real_images_2)
    else:
        real_images_2 = tf.image.flip_left_right(real_images_1)

    random_noise_1 = features['random_noise_1']    
    
    fake_images_out_1 = models.generator(random_noise_1, resolution, cfg, is_training=True)    
        
    real_scores_out = models.discriminator(real_images_1, resolution, cfg)
    fake_scores_out = models.discriminator(fake_images_out_1, resolution, cfg)
    #fake_scores_out_g = models.discriminator(fake_images_out_2, resolution, cfg)

    with tf.name_scope('Penalties'):
        d_loss = fake_scores_out - real_scores_out
        g_loss = -1.0 * fake_scores_out

        with tf.name_scope('GradientPenalty'):
            mixing_factors = tf.random_uniform([int(real_images_1.get_shape()[0]), 1, 1, 1], 0.0, 1.0, dtype=fake_images_out_1.dtype)
            mixed_images_out = ops.lerp(real_images_1, real_images_2, mixing_factors)
            mixed_scores_out = models.discriminator(mixed_images_out, resolution, cfg)
            mixed_loss = tf.reduce_sum(mixed_scores_out)
            mixed_grads = tf.gradients(mixed_loss, [mixed_images_out])[0]
            mixed_norms = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))        
            gradient_penalty = tf.square(mixed_norms - 1.0)
        d_loss += gradient_penalty * 10.0

        with tf.name_scope('EpsilonPenalty'):
            epsilon_penalty = tf.square(real_scores_out)
        d_loss += epsilon_penalty * 0.001

    resolution_step = utils.get_or_create_resolution_step()    
    fadein_rate = tf.minimum(tf.cast(resolution_step, tf.float32) / float(cfg.fadein_steps), 1.0)    
    learning_rate = cfg.base_learning_rate
    d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=cfg.beta1, beta2=cfg.beta2, epsilon=cfg.eps, name="AdamD")
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=cfg.beta1, beta2=cfg.beta2, epsilon=cfg.eps, name="AdamG")

    if cfg.data_format == 'NCHW':
        fake_images_out_1 = utils.nchw_to_nhwc(fake_images_out_1)
        real_images_1 = utils.nchw_to_nhwc(real_images_1)
        real_images_2 = utils.nchw_to_nhwc(real_images_2)
        mixed_images_out = utils.nchw_to_nhwc(mixed_images_out)
    tf.summary.image('generated_images', fake_images_out_1)
    tf.summary.image('real_images_1', real_images_1)
    tf.summary.image('real_images_2', real_images_2)
    tf.summary.image('mixed_images', mixed_images_out)
    with tf.variable_scope("Loss"):
        tf.summary.scalar('real_scores_out', tf.reduce_mean(real_scores_out))
        tf.summary.scalar('fake_scores_out', tf.reduce_mean(fake_scores_out))
        tf.summary.scalar('epsilon_penalty', tf.reduce_mean(epsilon_penalty))
        tf.summary.scalar('mixed_norms', tf.reduce_mean(mixed_norms))
    with tf.variable_scope("Rate"):
        tf.summary.scalar('fadein', fadein_rate)

    g_loss = tf.reduce_mean(g_loss)
    d_loss = tf.reduce_mean(d_loss)

    with tf.name_scope('TrainOps'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_step = d_optimizer.minimize(
                d_loss,
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='Discriminator'))
            g_step = g_optimizer.minimize(
                g_loss,
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='Generator'))
            with tf.control_dependencies([g_step, d_step]):       
                increment_global_step = tf.assign_add(
                    tf.train.get_or_create_global_step(), 1)
                increment_resolution_step = tf.assign_add(
                    utils.get_or_create_resolution_step(), 1)
            if resolution>=cfg.starting_resolution * 2:
                with tf.control_dependencies([increment_global_step, increment_resolution_step]):
                    lerp_ops = lerp_update_ops(resolution, fadein_rate)          
                    joint_op = tf.group([d_step, g_step, lerp_ops[0], lerp_ops[1], increment_global_step, increment_resolution_step])
            else:
                joint_op = tf.group([d_step, g_step, increment_global_step, increment_resolution_step])
    return joint_op, [g_loss, d_loss], [g_optimizer, d_optimizer]

def generate_step(cfg, resolution):
    graph = tf.Graph()
    restore_dir = os.path.join(cfg.model_dir, 'resolution_' + str(resolution))
    with graph.as_default(): # pylint: disable=E1129
        input = input_pipelines.PredictInputFunction(cfg.noise_dim, resolution)    
        params = {'data_dir' : cfg.data_dir, 'batch_size' : cfg.num_eval_images }
        features, labels = input(params)
        model = model_fn(features, labels, 'PREDICT', cfg)
        global_step = tf.train.get_or_create_global_step()
        with tf.Session() as sess:        
            sess.run(tf.global_variables_initializer())
            utils.restore(sess, restore_dir)
            images = sess.run(model)
            utils.write_images(images, cfg.model_dir+'/'+str(global_step.eval()).zfill(6)+'-'+str(resolution)+'.png', cfg.data_format)

    tf.reset_default_graph()

def train_step(cfg, resolution, restore_dir, store_dir):
    batch_size = cfg.resolution_to_batch_size[resolution]
    graph = tf.Graph()
    tf.gfile.MakeDirs(store_dir)
    ckpt_file = store_dir + '/model.ckp'
    global_step_value = 0    
    with graph.as_default(): # pylint: disable=E1129
        train_input = input_pipelines.TrainInputFunction(True, cfg.noise_dim, resolution, cfg.data_format)    
        params = {'data_dir' : cfg.data_dir, 'batch_size' : batch_size }
        features, labels = train_input(params)
        train_ops,[g_loss, d_loss],[g_optimizer, d_optimizer] = model_fn(features, labels, 'TRAIN', cfg)
        global_step = tf.train.get_or_create_global_step()        
        summary = tf.summary.merge_all()
        with tf.Session() as sess:                          
            sess.run(tf.global_variables_initializer())
            utils.restore(sess, restore_dir)
            saver = tf.train.Saver(name='main_saver')
            global_step_value = global_step.eval()
            if global_step_value == 0:
                utils.print_layers('Generator')
                utils.print_layers('Discriminator')
            if restore_dir != store_dir and restore_dir is not None:
                utils.print_layers('Generator')
                utils.print_layers('Discriminator')
                utils.reset_resolution_step()                
                sess.run(tf.variables_initializer(d_optimizer.variables()))
                sess.run(tf.variables_initializer(g_optimizer.variables()))
                saver.save(sess, ckpt_file, global_step = global_step)
            resolution_summary_writer = tf.summary.FileWriter(store_dir, sess.graph)                        
            start_time = time.time()            
            for _ in range(cfg.train_steps_before_eval // cfg.iterations_per_loop):
                start_time = time.time()
                for _ in trange(cfg.iterations_per_loop, leave=False):
                    sess.run(train_ops)
                    if global_step % cfg.resolution_steps == 0 and resolution != cfg.maximum_resolution:
                        break                        
                elapsed_time = time.time() - start_time
                g_loss_value, d_loss_value, global_step_value = sess.run([g_loss, d_loss, global_step])                
                tf.logging.info('Step %d - g_loss %f, d_loss %f, Sec/Step %f' % (global_step_value, g_loss_value, d_loss_value, elapsed_time / cfg.iterations_per_loop))
                summary_str = sess.run(summary)
                resolution_summary_writer.add_summary(summary_str, global_step_value)
                resolution_summary_writer.flush()
                if global_step % cfg.resolution_steps == 0 and resolution != cfg.maximum_resolution:
                    break                        
            global_step_value = global_step.eval()
            tf.logging.info('Saving parameters to %s' % (ckpt_file))            
            saver.save(sess, ckpt_file, global_step = global_step)                        
    tf.reset_default_graph()
    return global_step_value

def train(cfg):

    tf.gfile.MakeDirs(os.path.join(cfg.model_dir))        
    
    resolution = cfg.maximum_resolution    
    initial_checkpoint = None
    while initial_checkpoint is None and resolution != 1:
        restore_dir = os.path.join(cfg.model_dir, 'resolution_' + str(resolution))        
        initial_checkpoint = tf.train.latest_checkpoint(restore_dir)
        resolution = resolution // 2
    if initial_checkpoint is None or resolution == 1:
        resolution = cfg.starting_resolution    
        restore_dir = None
    else:
        resolution *= 2
        restore_dir = os.path.join(cfg.model_dir, 'resolution_' + str(resolution))
    tf.logging.info('Starting training for %d steps' % (cfg.train_steps))
    global_step = 0
    while global_step < cfg.train_steps:
        store_dir = os.path.join(cfg.model_dir, 'resolution_' + str(resolution))
        global_step = train_step(cfg, resolution, restore_dir, store_dir)
        restore_dir = store_dir
        tf.logging.info('Finished training for step %d' % (global_step))        
        generate_step(cfg, resolution)
        tf.logging.info('Finished generating images for step %d' % (global_step))
        if global_step % cfg.resolution_steps == 0 and resolution != cfg.maximum_resolution:
            resolution *= 2
            tf.logging.info('Change of resolution from %d to %d' % (resolution // 2, resolution))
            restore_dir = os.path.join(cfg.model_dir, 'resolution_' + str(resolution // 2))
            

def main(cfg):
    train(cfg)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    # Optimization hyperparams:
    parser.add_argument("--train_steps", type=int, default=500000,
                        help="Total train steps")
    parser.add_argument("--train_steps_before_eval", type=int, default=20 if DRY_RUN else 1000,
                        help="Train steps before evaluation")
    parser.add_argument("--resolution_steps", type=int, default= 30 if DRY_RUN else 10000,
                        help="Train steps for each resolution")
    parser.add_argument("--fadein_steps", type=int, default=29 if DRY_RUN else 8000,
                        help="Fadein steps for each resolution")
    parser.add_argument("--warmup_steps", type=int, default=5 if DRY_RUN else 800,
                        help="Warmup steps for learning rate")
    parser.add_argument("--iterations_per_loop", type=int, default=5 if DRY_RUN else 100,
                        help="Interations per loop")
    parser.add_argument("--num_eval_images", type=int, default=100,
                        help="Number of images for evaluation")
    parser.add_argument("--base_learning_rate", type=float, default=0.0005,
                        help="Base learning rate")
    parser.add_argument("--temperature", type=float, default=.9, help="temperature")                        
    parser.add_argument("--beta1", type=float, default=.0, help="beta1")    
    parser.add_argument("--beta2", type=float, default=.99, help="beta2")
    parser.add_argument("--eps", type=float, default=1e-6, help="eps")
    parser.add_argument("--report_histograms", type=bool, default=False,
                        help="If should report histograms")

    # Model hyperparams:
    parser.add_argument("--noise_dim", type=int, default=512,
                        help="Noise dimension")
    parser.add_argument("--starting_resolution", type=int, default=8,
                        help="Starting resolution")
    parser.add_argument("--maximum_resolution", type=int, default=128,
                        help="Maximum resolution")
    parser.add_argument("--data_format", type=str, default='NHWC',
                        help="Either NCHW or NHWC")

    # dataset
    parser.add_argument("--data_dir", type=str, default='C:/Projects/datasets/tfr-celeba128',
                        help="Bucket/Folder that contains the data tfrecord files")
    parser.add_argument("--model_dir", type=str, default='./output',
                        help="Output model directory")

    cfg = parser.parse_args()

    cfg.resolution_to_filt_num = {
        2: 512, 
        4: 512,
        8: 256,
        16: 256,
        32: 256,
        64: 128,
        128: 64
    }
    cfg.resolution_to_batch_size = {
        4: 128,
        8: 128,
        16: 128,
        32: 64,
        64: 64,
        128: 32
    }

    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'        
    tf.logging.set_verbosity(tf.logging.INFO)

    main(cfg)
