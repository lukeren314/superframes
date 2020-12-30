import numpy as np
import os, math, time, collections, numpy as np

import tensorflow as tf
from tensorflow.python.util import deprecation
import random as rn

import tensorflow.contrib.slim as slim
import sys, shutil, subprocess

from lib.ops import *
from lib.dataloader import inference_data_loader, frvsr_gpu_data_loader
from lib.frvsr import generator_F, fnet
from lib.Teco import FRVSR, TecoGAN

class SuperResolution:
    def __init__(self, checkpoint, sample_batch, cudaID = '0'):
        self.checkpoint = checkpoint
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        
        # fix all randomness, except for multi-treading or GPU process
        os.environ['PYTHONHASHSEED'] = '0'
        

        # Flags = tf.app.flags

        # Flags.DEFINE_integer('rand_seed', rand_seed , 'random seed' )

        # # Directories
        # Flags.DEFINE_string('input_dir_LR', input_dir_LR, 'The directory of the input resolution input data, for inference mode')
        # Flags.DEFINE_integer('input_dir_len', -1, 'length of the input for inference mode, -1 means all')
        # Flags.DEFINE_string('input_dir_HR', None, 'The directory of the input resolution input data, for inference mode')
        # Flags.DEFINE_string('mode', 'inference', 'train, or inference')
        # Flags.DEFINE_string('output_dir', output_dir, 'The output directory of the checkpoint')
        # Flags.DEFINE_string('output_pre', output_pre, 'The name of the subfolder for the images')
        # Flags.DEFINE_string('output_name', output_name, 'The pre name of the outputs')
        # Flags.DEFINE_string('output_ext', output_ext, 'The format of the output when evaluating')
        
        # # Models
        # Flags.DEFINE_string('checkpoint', checkpoint, 'If provided, the weight will be restored from the provided checkpoint')
        # Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')

        # FLAGS = Flags.FLAGS

        # Set CUDA devices correctly if you use multiple gpu system
        os.environ["CUDA_VISIBLE_DEVICES"]=cudaID
        
        FLAGS = TempConfig(sample_batch, -1, 16)

        # Declare the test data reader
        inference_data = inference_data_loader(FLAGS)
        input_shape = [1,] + list(inference_data.inputs[0].shape)
        output_shape = [1,input_shape[1]*4, input_shape[2]*4, 3]
        oh = input_shape[1] - input_shape[1]//8 * 8
        ow = input_shape[2] - input_shape[2]//8 * 8
        paddings = tf.constant([[0,0], [0,oh], [0,ow], [0,0]])

        # build the graph
        self.inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
        
        pre_inputs = tf.Variable(tf.zeros(input_shape), trainable=False, name='pre_inputs')
        pre_gen = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_gen')
        pre_warp = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_warp')
        
        transpose_pre = tf.space_to_depth(pre_warp, 4)
        inputs_all = tf.concat( (self.inputs_raw, transpose_pre), axis = -1)
        with tf.variable_scope('generator'):
            gen_output = generator_F(inputs_all, 3, reuse=False, FLAGS=FLAGS)
            # Deprocess the images outputed from the model, and assign things for next frame
            with tf.control_dependencies([ tf.assign(pre_inputs, self.inputs_raw)]):
                self.outputs = tf.assign(pre_gen, deprocess(gen_output))
        
        inputs_frames = tf.concat( (pre_inputs, self.inputs_raw), axis = -1)
        with tf.variable_scope('fnet'):
            gen_flow_lr = fnet( inputs_frames, reuse=False)
            gen_flow_lr = tf.pad(gen_flow_lr, paddings, "SYMMETRIC") 
            gen_flow = upscale_four(gen_flow_lr*4.0)
            gen_flow.set_shape( output_shape[:-1]+[2] )
        pre_warp_hi = tf.contrib.image.dense_image_warp(pre_gen, gen_flow)
        self.before_ops = tf.assign(pre_warp, pre_warp_hi)
        
        # In inference time, we only need to restore the weight of the generator
        var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='generator')
        var_list = var_list + tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
        
        self.weight_initiallizer = tf.train.Saver(var_list)
        
        # Define the initialization operation
        self.init_op = tf.global_variables_initializer()
        self.local_init_op = tf.local_variables_initializer()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True


    def enhance(self, input_dir_LR, output_dir, input_dir_len=-1, output_pre="", output_name="output", output_ext="png", rand_seed=1, num_resblock = 16):
        # python main.py --cudaID 0 --output_dir ./results/ --summary_dir ./results/log/ --mode inference --input_dir_LR ./LR/calendar --output_pre ./calendar --num_resblock 16 --checkpoint ./model/TecoGAN --output_ext png
        # Check the output directory to save the checkpoint
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Fix randomness
        rn.seed(rand_seed)
        np.random.seed(rand_seed)
        tf.set_random_seed(rand_seed)

        FLAGS = TempConfig(input_dir_LR, input_dir_len, num_resblock)

        # Declare the test data reader
        inference_data = inference_data_loader(FLAGS)
        if (output_pre == ""):
            image_dir = output_dir
        else:
            image_dir = os.path.join(output_dir, output_pre)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            
        with tf.Session(config=self.config) as sess:
            # Load the pretrained model
            sess.run(self.init_op)
            sess.run(self.local_init_op)
            
            print('Loading weights from ckpt model')
            self.weight_initiallizer.restore(sess, self.checkpoint)
            max_iter = len(inference_data.inputs)
                    
            srtime = 0
            print('Frame evaluation starts!!')
            for i in range(max_iter):
                input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
                feed_dict={self.inputs_raw: input_im}
                t0 = time.time()
                if(i != 0):
                    sess.run(self.before_ops, feed_dict=feed_dict)
                output_frame = sess.run(self.outputs, feed_dict=feed_dict)
                srtime += time.time()-t0
                
                if(i >= 5): 
                    name, _ = os.path.splitext(os.path.basename(str(inference_data.paths_LR[i])))
                    filename = output_name+'_'+name
                    print('saving image %s' % filename)
                    out_path = os.path.join(image_dir, "%s.%s"%(filename,output_ext))
                    frame = output_frame[0]
                    save_img(out_path, frame)
                else:# First 5 is a hard-coded symmetric frame padding, ignored but time added!
                    print("Warming up %d"%(5-i))
        print( "total time " + str(srtime) + ", frame number " + str(max_iter) )


class TempConfig:
    def __init__(self, input_dir_LR, input_dir_len, num_resblock):
        self.input_dir_LR = input_dir_LR
        self.input_dir_len = input_dir_len
        self.num_resblock = num_resblock