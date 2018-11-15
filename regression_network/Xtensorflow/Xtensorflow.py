import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages

from prototxt_basic import *
from helper import *
import numpy as np
import math

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'
XNET_VARIABLE_COLLECTION = 'xnet_varibale'
# xnet_collections = ['xnet_varibale',tf.GraphKeys.TRAINABLE_VARIABLES,tf.GraphKeys.GLOBAL_VARIABLES]
xnet_collections = ['xnet_varibale', tf.GraphKeys.GLOBAL_VARIABLES]
GLOBAL_EPS = 0.001
# ============================LAYER MAP==============================
layermap = {'Convolution': 'LAYER_CONVOLUTIONAL',
            'Pooling_MAX': 'LAYER_MAX_POOLING',
            'Pooling_AVE': 'LAYER_AVE_POOLING',
            'Pooling': 'LAYER_POOLING',
            'InnerProduct': 'LAYER_FULL_CONNECTION',
            'Concat': 'LAYER_CONCAT',
            'Dropout': 'LAYER_DROPOUT',
            'BatchNorm': 'LAYER_BN',
            'Scale': 'LAYER_SCALE',
            'LRN': 'LAYER_LRN',
            'Eltwise': 'LAYER_ELTWISE_SUM',
            'Deconvolution': 'LAYER_DECONVLUTIONAL',
            'Power': 'LAYER_POWER',
            'ReLU': 'LAYER_RELU'}

# ============================Xtensorflow==============================
class Xtensorflow():
    def __init__(self, input ,weight_decay = 0.0 ,is_train = True, model_name = 'load', load_from_txt = None, cnn_type = "CLASSIFY"):
        self.is_train = is_train
        self.load_from_txt = load_from_txt
        self.model_name = model_name
        self.layer_dict = {}
        self.input_dim = input.shape[-1]
        # self.input_w = input.shape[2]
        # self.input_h = input.shape[1]
        self.weight_decay = tf.constant(weight_decay, dtype=tf.float32)

        self.reduce_index = 0
        self.reduce_index_list = []

        layer = {}
        layer['index'] = 0
        layer['output'] = input
        layer['reduce_index'] = 0
        self.reduce_index_list.append(self.reduce_index)
        self.layer_dict[str(0)] = layer
        self.cnn_type = cnn_type


    def layer_info(self, index, input_index, input_num, layer_type,  input_dim, output_dim,
                   kernel_size, stride, padding, dialation, bias, activation, layer_output, reduce_index):

        layer = {}
        layer['index'] = index
        layer['input_index'] = input_index
        layer['input_num'] = input_num
        layer['layer_type'] = layer_type
        layer['input_dim'] = input_dim
        layer['output_dim'] = output_dim
        layer['kernel_size'] = kernel_size
        layer['stride'] = stride
        layer['padding'] = padding
        layer['dialation'] = dialation
        layer['bias'] = bias
        layer['output'] = layer_output
        layer['reduce_index'] = reduce_index

        if layer_type == 'LAYER_CONVOLUTIONAL':
            layer['name'] = 'conv_'+str(index)
        elif layer_type == 'LAYER_CONCAT':
            layer['name'] = 'concat_' + str(index)
        elif layer_type == 'LAYER_NN_RESIZE':
            layer['name'] = 'resize_' + str(index)

        self.reduce_index_list.append(reduce_index)
        if activation == tf.nn.relu:
            layer['activation'] = 'ACTIVE_RECTIFIED_LINEAR'
        elif activation == tf.nn.tanh:
            layer['activation'] = 'ACTIVE_TANH'
        elif activation == None:
            layer['activation'] = 'ACTIVE_LINEAR'

        return layer


    def conv_layer(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_Conv_"+str(index)) as scope:

            input = self.get_layer_output(input_index)

            input_dim = input.shape[-1]
            output_dim = output_shape[-1]

            input_h = int(input.shape[1])
            input_w = int(input.shape[2])

            if input_h%stride !=0 or input_w%stride !=0:
                print('Conv Input Stride Error')
                return

            output_h = input_h / stride
            output_w = input_w / stride

            kernel_size_tmp = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_h = int(math.ceil(((output_h-1)*stride + kernel_size_tmp - input_h)/2.0))
            pad_w = int(math.ceil(((output_w-1)*stride + kernel_size_tmp - input_w)/2.0))

            #Note: tensorflow "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right
            #      Xnet, Caffe pad left first,then img2col
            if pad_w > 0:
                input = tf.pad(input, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], "CONSTANT")

            if self.load_from_txt !=None:
                weights_path = self.load_from_txt + '/model/'+ str(index) + '/weights.txt'
                biases_path = self.load_from_txt + '/model/'+ str(index) + '/biases.txt'

                w = read_from_txt(weights_path)
                b = read_from_txt(biases_path)

                print(w.shape)
                print(int(output_dim)*int(input_dim)*int(kernel_size)*int(kernel_size))
                # w = np.loadtxt(weights_path, delimiter=' ')
                    # .astype(np.float32)
                # b = np.loadtxt(biases_path, delimiter=' ')
                    # .astype(np.float32)

                w = w.reshape([int(output_dim), input_dim, kernel_size, kernel_size])
                w = w.transpose(2, 3, 1, 0)

                weights = tf.get_variable('weights',initializer=w,
                                          regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                          collections=xnet_collections)
                biases = tf.get_variable('biases',
                                         initializer=b,collections=xnet_collections)
            else:
                weights = tf.get_variable('weights', [kernel_size, kernel_size, input.get_shape()[-1],output_shape[-1] ],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),collections=xnet_collections)
                biases = tf.get_variable('biases', [output_shape[-1]],
                                         initializer=tf.constant_initializer(0.0),collections=xnet_collections)

            if rate == 1:
                conv = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='VALID')
            else:
                conv = tf.nn.atrous_conv2d(input, weights, rate, padding = 'VALID')

            # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            conv = tf.nn.bias_add(conv, biases)

            if activation != None:
                conv = activation(conv)

            print(conv)
            layer = self.layer_info(index, input_index, 1, 'LAYER_CONVOLUTIONAL',
                                    input.shape[3], output_shape[-1], kernel_size, stride, pad_w, rate, True, activation, conv, self.reduce_index)

            self.layer_dict[str(index)] = layer
            return index

    def conv_layer_v2(self , input_index, output_shape, kernel_size, stride, is_bias, activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_Conv_"+str(index)) as scope:

            input = self.get_layer_output(input_index)

            input_dim = input.shape[-1]
            output_dim = output_shape[-1]

            input_h = int(input.shape[1])
            input_w = int(input.shape[2])

            if input_h%stride !=0 or input_w%stride !=0:
                print('Conv Input Stride Error')
                return

            output_h = input_h / stride
            output_w = input_w / stride

            kernel_size_tmp = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_h = int(math.ceil(((output_h-1)*stride + kernel_size_tmp - input_h)/2.0))
            pad_w = int(math.ceil(((output_w-1)*stride + kernel_size_tmp - input_w)/2.0))

            #Note: tensorflow "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right
            #      Xnet, Caffe pad left first,then img2col
            if pad_w > 0:
                input = tf.pad(input, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], "CONSTANT")


            weights = tf.get_variable('weights', [kernel_size, kernel_size, input.get_shape()[-1],output_shape[-1] ],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),collections=xnet_collections)

            if rate == 1:
                conv = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='VALID')
            else:
                conv = tf.nn.atrous_conv2d(input, weights, rate, padding = 'VALID')

            # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            if is_bias:
                biases = tf.get_variable('biases', [output_shape[-1]],
                                             initializer=tf.constant_initializer(0.0),collections=xnet_collections)
                conv = tf.nn.bias_add(conv, biases)

            if activation != None:
                conv = activation(conv)

            print(conv)
            layer = self.layer_info(index, input_index, 1, 'LAYER_CONVOLUTIONAL',
                                    input.shape[3], output_shape[-1], kernel_size, stride, pad_w, rate, is_bias, activation, conv, self.reduce_index)

            self.layer_dict[str(index)] = layer
            return index

    def scale_layer(self , input_index, activation, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_Scale_"+str(index)) as scope:
            input = self.get_layer_output(input_index)
            inputs_shape = input.get_shape()

            if self.load_from_txt !=None:
                weights_path = self.load_from_txt + '/model/'+ str(index) + '/weights.txt'
                biases_path = self.load_from_txt + '/model/'+ str(index) + '/biases.txt'

                w = read_from_txt(weights_path)
                b = read_from_txt(biases_path)

                print(w.shape)

                w = w.reshape([inputs_shape[-1]])
                b = b.reshape([inputs_shape[-1]])


                gamma = tf.get_variable('gamma',initializer=w, collections=xnet_collections)

                beta = tf.get_variable('beta', initializer=b, collections=xnet_collections)
            else:

                gamma = tf.get_variable('gamma', inputs_shape[-1], initializer=tf.ones_initializer(),
                                            collections=xnet_collections)

                beta = tf.get_variable('beta', inputs_shape[-1], initializer=tf.zeros_initializer(),
                                           collections=xnet_collections)

            out = tf.nn.bias_add(input * gamma, beta)
            if activation != None:
                out = activation(out)

            print(out)
            layer = self.layer_info(index, input_index, 1, 'LAYER_SCALE',
                                    input.shape[3], inputs_shape[-1], 0, 0, 0, 0, True, activation, out, self.reduce_index)



            self.layer_dict[str(index)] = layer
            return index

    def depthwise_conv_layer(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_GroupConv_"+str(index)) as scope:

            input = self.get_layer_output(input_index)

            input_dim = int(input.shape[-1])
            output_dim = int(output_shape[-1])

            if input_dim != output_dim:
                print('group dim error!!!')
                return

            input_h = int(input.shape[1])
            input_w = int(input.shape[2])
            # output_h = int(output_shape[1])
            # output_w = int(output_shape[2])

            if input_h%stride !=0 or input_w%stride !=0:
                print('Conv Input Stride Error')
                return

            output_h = input_h / stride
            output_w = input_w / stride

            kernel_size_tmp = kernel_size
            pad_h = int(math.ceil(((output_h-1)*stride + kernel_size_tmp - input_h)/2.0))
            pad_w = int(math.ceil(((output_w-1)*stride + kernel_size_tmp - input_w)/2.0))

            #Note: tensorflow "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right
            #      Xnet,Caffe pad left first,then img2col
            if pad_w > 0:
                input = tf.pad(input, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], "CONSTANT")

            depthwise_shape = [kernel_size, kernel_size, input_dim, 1]

            depthwise_weights = tf.get_variable('weights', depthwise_shape,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      collections=xnet_collections)
            depthwise_biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0),collections=xnet_collections)

            conv = tf.nn.depthwise_conv2d(input, depthwise_weights, [1, stride, stride, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, depthwise_biases)

            if activation != None:
                conv = activation(conv)

            print(conv)

            layer = self.layer_info(index, input_index, 1, 'LAYER_CONVOLUTIONAL',
                                    input.shape[3], output_shape[-1], kernel_size, stride, pad_w, rate, True, activation, conv, self.reduce_index)


            self.layer_dict[str(index)] = layer
            return index

    def true_sub_pixel_conv_layer(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_Conv_"+str(index)) as scope:

            input = self.get_layer_output(input_index)

            input_dim = input.shape[-1]
            output_dim = output_shape[-1]* (rate**2)

            input_h = int(input.shape[1])
            input_w = int(input.shape[2])
            output_h = int(output_shape[1] / rate)
            output_w = int(output_shape[2] / rate)



            kernel_size_tmp = kernel_size + (kernel_size - 1) * (1 - 1)
            pad_h = int(math.ceil(((output_h-1)*stride + kernel_size_tmp - input_h)/2.0))
            pad_w = int(math.ceil(((output_w-1)*stride + kernel_size_tmp - input_w)/2.0))

            #Note: tensorflow "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right
            #      Xnet,Caffe pad left first,then img2col
            if pad_w > 0:
                input = tf.pad(input, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], "CONSTANT")

            ww = 0.02 * np.random.random(size=kernel_size * kernel_size) - 0.01
            # ww =  np.ones([kernel_size * kernel_size]) / (kernel_size*kernel_size)

            ww_list = []
            for i in range(input_dim*output_dim):
                ww_list.append(ww)

            ww = np.asarray(ww_list)
            ww = ww.reshape([input_dim, output_dim, kernel_size, kernel_size])
            ww = ww.transpose(2, 3, 0, 1)
            ww = ww.astype(np.float32)

            weights = tf.get_variable('weights', initializer=ww,
                                    regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),collections=xnet_collections)
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0),collections=xnet_collections)

            conv = tf.nn.atrous_conv2d(input, weights, 1, padding = 'VALID')
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

            def _phase_shift(I, r):
                bsize, a, b, c = I.get_shape().as_list()
                bsize = tf.shape(I)[0]
                X = tf.reshape(I, (bsize, a, b, r, r))
                X = tf.transpose(X, (0, 1, 2, 4, 3))
                X = tf.split(X, a, 1)
                X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
                X = tf.split(X, b, 1)
                X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
                return tf.reshape(X, (bsize, a * r, b * r, 1))

            def _PS(X, r, n_out_channel):
                if n_out_channel > 1:
                    assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel
                    Xc = tf.split(X, n_out_channel, 3)
                    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
                elif n_out_channel == 1:
                    X = _phase_shift(X, r)
                return X


            n_out_channel = output_shape[-1]
            sub_pixel_conv = _PS(conv, r=rate, n_out_channel=n_out_channel)

            if activation != None:
                sub_pixel_conv = activation(sub_pixel_conv)

            layer = self.layer_info(index, input_index, 1, 'LAYER_SUBPIXELCONV',
                                    input.shape[3], output_shape[3],  kernel_size, stride, pad_w, rate, True, activation, sub_pixel_conv, self.reduce_index)

            self.layer_dict[str(index)] = layer
            return index

    def sub_pixel_conv_layer(self , input_index, output_shape, rate, activation, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_Conv_"+str(index)) as scope:

            input = self.get_layer_output(input_index)

            def _phase_shift(I, r):
                bsize, a, b, c = I.get_shape().as_list()
                bsize = tf.shape(I)[0]
                X = tf.reshape(I, (bsize, a, b, r, r))
                X = tf.transpose(X, (0, 1, 2, 4, 3))
                X = tf.split(X, a, 1)
                X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
                X = tf.split(X, b, 1)
                X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
                return tf.reshape(X, (bsize, a * r, b * r, 1))

            def _PS(X, r, n_out_channel):
                if n_out_channel > 1:
                    assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel
                    Xc = tf.split(X, n_out_channel, 3)
                    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
                elif n_out_channel == 1:
                    X = _phase_shift(X, r)
                return X


            n_out_channel = int(int(input.get_shape()[-1]) / (rate ** 2))

            sub_pixel_conv = _PS(input, r=rate, n_out_channel=n_out_channel)

            if activation != None:
                sub_pixel_conv = activation(sub_pixel_conv)


            layer = self.layer_info(index, input_index, 1, 'LAYER_SUBPIXELCONV',
                                    input.shape[3], output_shape[3], 0, 0, 0, rate, True, activation, sub_pixel_conv, self.reduce_index)

            self.layer_dict[str(index)] = layer
            return index

    def deconv_layer(self , input_index, output_shape, kernel_size, stride , activation, index = None):

        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_Deconv_" + str(index)) as scope:

            input = self.get_layer_output(input_index)

            output_w = int(input.shape[2])
            input_w = int(output_shape[2])
            pad = int(((output_w-1)*stride + kernel_size - input_w)/2.0)

            w = tf.get_variable('weights', [kernel_size, kernel_size, output_shape[-1], input.get_shape()[-1]],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),collections=xnet_collections)
            deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1, stride, stride, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0),collections=xnet_collections)
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            if activation != None:
                deconv = activation(deconv)

            layer = self.layer_info(index, input_index, 1, 'LAYER_DECONVLUTIONAL',
                                    input.shape[3], output_shape[3], kernel_size, stride, pad, 1, True, activation, deconv)

            self.layer_dict[str(index)] = layer

            return index

    def resize_layer(self, input_index, output_shape, activation = None, index=None):

            if index == None:
                index = input_index + 1

            with tf.variable_scope("Layer_Resize_" + str(index)) as scope:

                input = self.get_layer_output(input_index)

                output_w = int(output_shape[2])
                output_h = int(output_shape[1])


                output = tf.image.resize_images(input, [output_h, output_w], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                if activation != None:
                    output = activation(output)

                # self.reduce_index += 1
                layer = self.layer_info(index, input_index, 1, 'LAYER_NN_RESIZE',
                                        input.shape[3], output_shape[-1], 0, 2, 0, 0, False, activation,
                                        output, self.reduce_index)

                self.layer_dict[str(index)] = layer

                return index

    def bn_layer(self , input_index, activation, index = None):
        # batch_norm layer
        # bn_layer in Xnet

        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_BatchNorm_" + str(index)) as scope:

            input = self.get_layer_output(input_index)

            def batch_norm(inputs, decay=0.99, epsilon=0.001, activation=None):

                inputs_shape = inputs.get_shape()
                axis = list(range(len(inputs_shape) - 1))
                params_shape = inputs_shape[-1:]

                # Create moving_mean and moving_variance add them to
                # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
                moving_collections = xnet_collections#, tf.GraphKeys.MOVING_AVERAGE_VARIABLES
                moving_mean = tf.get_variable('moving_mean', params_shape,
                                              initializer=tf.zeros_initializer(),
                                              trainable=False,
                                              collections=moving_collections)
                moving_variance = tf.get_variable('moving_variance', params_shape,
                                                  initializer=tf.ones_initializer(),
                                                  trainable=False,
                                                  collections=moving_collections)


                # Calculate the moments based on the individual batch.
                mean, variance = tf.nn.moments(inputs, axis)
                update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay, zero_debias = False)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay, zero_debias = False)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

                def mean_var_with_update():
                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(mean), tf.identity(variance)

                # Allocate parameters for the beta and gamma of the normalization.
                beta, gamma = None, None

                if self.is_train:
                    mean, var = mean_var_with_update()
                    outputs = tf.identity(tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon))
                else:
                    outputs = tf.identity(tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon))

                outputs.set_shape(inputs.get_shape())
                if activation:
                    outputs = activation(outputs)
                return outputs

            output = batch_norm(input, activation=activation)

            # ============================LAYER_BN=================================
            self.reduce_index += 1
            layer = self.layer_info(index, input_index, 1, 'LAYER_BN',
                                    int(input.shape[-1]), int(input.shape[-1]), 0, 0, 0, 0, False, None, output, self.reduce_index)
            self.layer_dict[str(index)] = layer

        return index

    def bn_with_scale_layer(self , input_index, activation, index = None):
        # batch_norm layer with scale and offset
        # bn_layer + scale_layer in Xnet

        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_BatchNorm_" + str(index)) as scope:

            input = self.get_layer_output(input_index)

            def batch_norm(inputs, decay=0.99, center=True, scale=True, epsilon=0.001,
                           activation=None):

                inputs_shape = inputs.get_shape()
                axis = list(range(len(inputs_shape) - 1))
                params_shape = inputs_shape[-1:]

                # Create moving_mean and moving_variance add them to
                # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
                moving_collections = xnet_collections#, tf.GraphKeys.MOVING_AVERAGE_VARIABLES
                moving_mean = tf.get_variable('moving_mean', params_shape,
                                              initializer=tf.zeros_initializer(),
                                              trainable=False,
                                              collections=moving_collections)
                moving_variance = tf.get_variable('moving_variance', params_shape,
                                                  initializer=tf.ones_initializer(),
                                                  trainable=False,
                                                  collections=moving_collections)


                # Calculate the moments based on the individual batch.
                mean, variance = tf.nn.moments(inputs, axis)
                update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay, zero_debias = False)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay, zero_debias = False)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

                def mean_var_with_update():
                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(mean), tf.identity(variance)

                # Allocate parameters for the beta and gamma of the normalization.
                beta, gamma = None, None

                if scale:
                    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer(),collections=xnet_collections)

                if center:
                    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer(),collections=xnet_collections )
                # Normalize the activations.


                if self.is_train:
                    mean, var = mean_var_with_update()
                    outputs = tf.identity(tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon))
                else:
                    outputs = tf.identity(tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon))

                outputs.set_shape(inputs.get_shape())
                if activation:
                    outputs = activation(outputs)
                return outputs

            # batch_norm()
            output = batch_norm(input, activation=activation)

            # ============================LAYER_BN=================================
            self.reduce_index += 1
            layer = self.layer_info(index, input_index, 1, 'LAYER_BN',
                                    int(input.shape[-1]), int(input.shape[-1]), 0, 0, 0, 0, False, None, output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            # ============================LAYER_SCALE==============================
            self.reduce_index += 1
            layer = self.layer_info(index+1, index, 1, 'LAYER_SCALE',
                                    int(input.shape[-1]), int(input.shape[-1]), 0, 0, 0, 0, False, activation, output, self.reduce_index)
            self.layer_dict[str(index+1)] = layer

        return index+1

    def conv_with_bn_layer(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        conv_index = self.conv_layer(input_index, output_shape, kernel_size, stride , None , rate = rate, index = index)
        bn_index = self.bn_with_scale_layer(conv_index, activation)

        return bn_index

    def conv_with_bn_layer_v2(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        conv_index = self.conv_layer_v2(input_index, output_shape, kernel_size, stride , False, None, rate = rate,  index = index)
        bn_index = self.bn_with_scale_layer(conv_index, activation)

        return bn_index

    def bn_with_conv_layer(self , input_index, output_shape, kernel_size, stride , activation, index = None):
        # conv layer with bn
        # conv_layer + bn_layer + scale_layer in Xnet
        if index == None:
            index = input_index + 1

        bn_index = self.bn_with_scale_layer(input_index, activation)
        conv_index = self.conv_layer(bn_index, output_shape, kernel_size, stride , None)

        return conv_index

    def depthwize_conv_with_bn_layer(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        conv_index = self.depthwise_conv_layer(input_index, output_shape, kernel_size, stride , None , rate = rate, index = index)
        bn_index = self.bn_with_scale_layer(conv_index, activation)

        return bn_index

    def conv_with_se_layer(self , input_index, output_shape, kernel_size, stride , activation, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        conv_index = self.conv_layer(input_index, output_shape, kernel_size, stride , activation , rate = 1, index = index)
        se_index = self.squeeze_excitation_layer(conv_index)

        return se_index

    def deconv_with_bn_layer(self , input_index, output_shape, kernel_size, stride , activation, index = None):
        # deconv layer with bn
        # deconv_layer + bn_layer + scale_layer in Xnet
        if index == None:
            index = input_index + 1

        deconv_index = self.deconv_layer(input_index, output_shape, kernel_size, stride , None)
        bn_index = self.bn_with_scale_layer( deconv_index, activation)

        return bn_index

    def concat_layer(self ,input_index_list,index):
        # concat op
        # concat_layer in Xnet
        with tf.variable_scope("Layer_Concat_" + str(index)) as scope:

            tensor_list = [self.get_layer_output(layerindex ) for layerindex in input_index_list]
            dim_list = [int(tensor.shape[-1]) for tensor in tensor_list]

            output = tf.concat(values = tensor_list, axis = 3)

            print(output)

            layer = self.layer_info(index, input_index_list, 1, 'LAYER_CONCAT',
                                    dim_list, int(output.shape[-1]), 0, 0, 0, 0, False, None, output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            return index

    def sum_layer(self, input_index_list, index ,activation = None):
        # elementwise sum op
        # concat_layer in Xnet
        with tf.variable_scope("Layer_Sum_" + str(index)) as scope:
            tensor_list = [self.get_layer_output(layerindex) for layerindex in input_index_list]
            dim_list = [int(tensor.shape[-1]) for tensor in tensor_list]
            output = tensor_list[0]
            for i in range(1,len(tensor_list)):
                # if output.shape != tensor_list[i].shape:
                #     print('Sum op Wrong Shape')
                #     return
                output = tf.add(output, tensor_list[i])

            if activation != None:
                output = activation(output)
            print(output)

            # dim_list = [int(tensor.shape[-1]) for tensor in tensor_list]
            layer = self.layer_info(index, input_index_list, 1, 'LAYER_ELTWISE_SUM',
                                    dim_list, int(output.shape[-1]), 0, 0, 0, 0, False, activation, output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            return index

    def maxpooling_layer(self , input_index, output_shape, kernel_size, stride ,padding = 'VALID', activation = None, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_MaxPooling_"+str(index)) as scope:
            input = self.get_layer_output(input_index)


            if padding == 'VALID':
                input_h = int(input.shape[1])
                input_w = int(input.shape[2])

                if input_h % stride != 0 or input_w % stride != 0:
                    print('Conv Input Stride Error')
                    return

                # output_h = math.ceil(input_h / stride)
                # output_w = math.ceil(input_w / stride)

                pad_h = kernel_size - input_h%stride - stride
                pad_w = kernel_size - input_w%stride - stride

                if pad_w > 0:
                    input = tf.pad(input, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]], "CONSTANT")

                output = tf.nn.max_pool(input, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding)
            else:
                output = tf.nn.max_pool(input, [1,kernel_size,kernel_size,1], [1,stride,stride,1], padding)

            if activation != None:
                output = activation(output)

            print(output)

            layer = self.layer_info(index, input_index, 1, 'LAYER_MAX_POOLING',
                                    input.shape[3], output_shape[-1], kernel_size, stride, 0, 0, False,activation, output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            return index

    def avepooling_layer(self, input_index, output_shape, kernel_size, stride, padding = 'VALID' ,activation = None, index=None, dropout = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_AvePooling_" + str(index)) as scope:
            input = self.get_layer_output(input_index)
            output = tf.nn.avg_pool(input, [1,kernel_size,kernel_size,1], [1,stride,stride,1], padding)

            if activation != None:
                output = activation(output)

            if dropout != None:
                output = tf.nn.dropout(output,dropout)

            print(output)

            layer = self.layer_info(index, input_index, 1, 'LAYER_AVE_POOLING',
                                    input.shape[3], output_shape[-1], kernel_size, stride, 0, 0, False, activation,
                                    output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            return index

    def fc_layer(self, input_index, output_shape, activation = None, index=None, dropout = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_FC_" + str(index)) as scope:
            input = self.get_layer_output(input_index)
            shape = input.get_shape().as_list()
            input_dim = 1
            for d in shape[1:]:
                input_dim *= d

            # input = tf.reshape(input, [-1, shape[-1],shape[1],shape[2]])
            if len(shape) == 4:
                input = tf.transpose(input,[0,3,1,2])
            input = tf.reshape(input, [-1, input_dim])
            output_dim = output_shape[1]

            if self.load_from_txt != None:
                weights_path = self.load_from_txt + '/model/'+ str(index) + '/weights.txt'
                biases_path = self.load_from_txt + '/model/'+ str(index) + '/biases.txt'

                # w = np.loadtxt(weights_path, delimiter=' ',dtype=float).astype(np.float32)
                # b = np.loadtxt(biases_path, delimiter=' ', dtype=float).astype(np.float32)

                w = read_from_txt(weights_path)
                b = read_from_txt(biases_path)

                w = w.reshape([output_dim, input_dim])
                w = w.transpose(1, 0)

                weights = tf.get_variable('weights',initializer=w,
                                            regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),collections=xnet_collections)
                biases = tf.get_variable('biases', initializer=b,collections=xnet_collections)

            else:
                weights = tf.get_variable('weights', [input_dim, output_shape[1]],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),collections=xnet_collections)
                biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0),
                                         collections=xnet_collections)

            output = tf.nn.xw_plus_b(input, weights, biases)

            if activation != None:
                output = activation(output)

            if dropout != None:
                output = tf.nn.dropout(output,dropout)

            print(output)
            layer = self.layer_info(index, input_index, 1, 'LAYER_FULL_CONNECTION',
                                    input.shape[-1], output_shape[-1], 1, 1, 0, 0, True, activation,
                                    output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            return index

    def reshape_layer(self, input_index, output_shape, activation = None, index=None):
        if index == None:
            index = input_index + 1

            input = self.get_layer_output(input_index)
            output = tf.reshape(input, output_shape)

            if activation != None:
                output = activation(output)

            print(output)
            layer = self.layer_info(index, input_index, 1, 'LAYER_RESHAPE',
                                    input.shape[-1], output_shape[-1], 1, 1, 0, 0, False, activation,
                                    output, self.reduce_index)
            self.layer_dict[str(index)] = layer

            return index

    def resnet_block_layer(self , input_index, output_shape, kernel_size, stride , activation, index = None):
        # conv layer with bn
        # conv_layer + bn_layer + scale_layer in Xnet

        conv_index0 = self.conv_layer(input_index, output_shape, kernel_size, stride , tf.nn.relu )
        conv_index1 = self.conv_layer(conv_index0, output_shape, kernel_size, stride , None)
        sum_index = self.sum_layer([input_index , conv_index1], conv_index1 + 1, activation )

        return sum_index

    def squeeze_excitation_layer(self , input_index, activation = None, rate = 1, index = None):
        if index == None:
            index = input_index + 1

        with tf.variable_scope("Layer_SqueezeExcitation_"+str(index)) as scope:
            input = self.get_layer_output(input_index)

            input_dim = int(input.shape[-1])
            input_h = int(input.shape[1])
            input_w = int(input.shape[2])

            global_pooling = tf.nn.avg_pool(input, [1, input_h, input_w, 1], [1, 1, 1, 1], 'VALID')
            #==============================squeeze excitation==========================================================
            squeeze = tf.reshape(global_pooling, [-1, input_dim])

            weights_0 = tf.get_variable('weights_0', [input_dim, input_dim*rate],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      collections=xnet_collections)

            biases_0 = tf.get_variable('biases_0', [input_dim*rate], initializer=tf.constant_initializer(0.0),
                                     collections=xnet_collections)

            excitation = tf.nn.relu( tf.nn.xw_plus_b(squeeze, weights_0, biases_0) )

            weights_1 = tf.get_variable('weights_1', [input_dim*rate, input_dim],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      collections=xnet_collections)

            biases_1 = tf.get_variable('biases_1', [input_dim], initializer=tf.constant_initializer(0.0),
                                     collections=xnet_collections)

            excitation = tf.nn.sigmoid( tf.nn.xw_plus_b(excitation, weights_1, biases_1) )
            excitation = tf.reshape(excitation, [-1,1,1,input_dim])

            output = input * excitation
            # ==============================layer info==========================================================
            layer = self.layer_info(index, input_index, 1, 'LAYER_SQUEEZEEXCITATION',
                                    input.shape[3], input.shape[3], 0, 0, 0, 0, True, activation, output, self.reduce_index)

            self.layer_dict[str(index)] = layer

            return index

    def get_layer_output(self,index):
        layer = self.layer_dict[str(index)]
        return layer['output']

    def get_network_output(self):
        num = len(self.layer_dict)
        layer = self.layer_dict[str(num-1)]
        return layer['output']

    def create_xnetlist(self, sess):
        weight_list = []
        bias_list = []

        # model_variable = sess._graph._collections['xnet_variable']
        model_variables = tf.get_collection('xnet_varibale')
        model_variables = [var for var in model_variables if self.model_name in var.name]

        index = 0
        while index < len(model_variables):

            if 'Conv' in model_variables[index].name:
                print(model_variables[index].name)
                print(model_variables[index + 1].name)
                conv_w_tmp = sess.run(model_variables[index])
                conv_b_tmp = sess.run(model_variables[index + 1])
                conv_w_tmp = conv_w_tmp.transpose(3, 2, 0, 1)
                conv_b_tmp = conv_b_tmp.reshape([-1])

                index += 2
                # v_tmp =  model_variables[index].name
                if index < len(model_variables) and model_variables[index].name.find('BatchNorm')>0:
                    print(model_variables[index].name)
                    print(model_variables[index+1].name)
                    print(model_variables[index+2].name)
                    print(model_variables[index+3].name)

                    bn_mean_tmp = sess.run(model_variables[index])
                    bn_variance_tmp = sess.run(model_variables[index + 1])
                    bn_gamma_tmp = sess.run(model_variables[index + 2])
                    bn_beta_tmp = sess.run(model_variables[index + 3])

                    bn_variance_tmp = np.sqrt(bn_variance_tmp.reshape([-1]) + 0.001)
                    bn_mean_tmp = bn_mean_tmp.reshape([-1])
                    bn_gamma_tmp = bn_gamma_tmp.reshape([-1])
                    bn_beta_tmp = bn_beta_tmp.reshape([-1])

                    index += 4
                    fmap_num  = conv_w_tmp.shape[0]
                    for f_index in range(fmap_num):
                        conv_w_tmp[f_index,:,:,:] = conv_w_tmp[f_index,:,:,:] * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index]
                        conv_b_tmp[f_index] = (conv_b_tmp[f_index] - bn_mean_tmp[f_index]) * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index] + bn_beta_tmp[f_index]

                    weight_list.append(conv_w_tmp.reshape([-1]))
                    bias_list.append(conv_b_tmp)
                else:
                    weight_list.append(conv_w_tmp.reshape([-1]))
                    bias_list.append(conv_b_tmp)

            if index < len(model_variables) and 'FC' in model_variables[index].name:
                print(model_variables[index].name)
                print(model_variables[index + 1].name)

                w_tmp = sess.run(model_variables[index])
                b_tmp = sess.run(model_variables[index + 1])

                w_tmp = w_tmp.transpose(1, 0)
                weight_list.append(w_tmp.reshape([-1]))
                bias_list.append(b_tmp.reshape([-1]))

                index += 2

        layer_list = []
        for i in range(1,len(self.layer_dict)):
            layer = self.layer_dict[str(i)]

            if layer['layer_type'] == 'LAYER_BN' or layer['layer_type'] == 'LAYER_SCALE':
                # TODO
                todo = []
            # elif layer['layer_type'] == 'LAYER_NN_RESIZE':
            #     #do nothing
            #     index_list = layer['input_index']
            #     dim_list = layer['input_dim']
            #     str_list = [str(layer['index'] - layer['reduce_index']),
            #                 str(index_list[index] - self.reduce_index_list[index_list[index]]), str(len(index_list)),
            #                 layer['layer_type'],
            #                 str(dim_list[index]), str(layer['outpt_dim']), '0', '0', '0', '0', 'False',
            #                 layer['activation']]

            elif layer['layer_type'] == 'LAYER_CONCAT' or layer['layer_type'] == 'LAYER_ELTWISE_SUM':
                index_list = layer['input_index']
                dim_list = layer['input_dim']
                for index in range(len(index_list)):
                    str_list = [str(layer['index']- layer['reduce_index']),str(index_list[index]- self.reduce_index_list[index_list[index]]),str(len(index_list)),layer['layer_type'],
                                str(dim_list[index]),str(layer['output_dim']),'0','0','0','0','False', layer['activation']]
                    layer_list.append(str_list)

            else:
                str_list = [str(layer['index']- layer['reduce_index']), str(layer['input_index']- self.reduce_index_list[layer['input_index']]), str(layer['input_num']),layer['layer_type'],
                            str(layer['input_dim']),str(layer['output_dim']),str(layer['kernel_size']),
                            str(layer['stride']), str(layer['padding']), str(layer['dialation']),str(layer['bias']),
                            layer['activation']]
                layer_list.append(str_list)
        return layer_list, weight_list, bias_list

    def create_xnetlist_v2(self, sess):
        layer_list = []
        for i in range(1, len(self.layer_dict)):
            layer = self.layer_dict[str(i)]
            if layer['layer_type'] == 'LAYER_SCALE':
                # TODO
                todo = []
            elif layer['layer_type'] == 'LAYER_BN':
                layer_list.append(layer)
            elif layer['layer_type'] == 'LAYER_CONCAT' or layer['layer_type'] == 'LAYER_ELTWISE_SUM':
                layer_list.append(layer)
            else:
                layer_list.append(layer)

        weight_list = []
        bias_list = []

        # model_variable = sess._graph._collections['xnet_variable']
        model_variables = tf.get_collection('xnet_varibale')
        model_variables = [var for var in model_variables if self.model_name in var.name]

        index = 0
        while index < len(model_variables):
            name = model_variables[index].name
            # layer_index
            if 'Conv' in model_variables[index].name:
                print(model_variables[index].name)
                print(model_variables[index + 1].name)
                conv_w_tmp = sess.run(model_variables[index])
                conv_b_tmp = sess.run(model_variables[index + 1])
                conv_w_tmp = conv_w_tmp.transpose(3, 2, 0, 1)
                conv_b_tmp = conv_b_tmp.reshape([-1])

                index += 2
                # v_tmp =  model_variables[index].name
                if index < len(model_variables) and model_variables[index].name.find('BatchNorm')>0:
                    # print(model_variables[index].name)
                    # print(model_variables[index+1].name)
                    # print(model_variables[index+2].name)
                    # print(model_variables[index+3].name)

                    bn_mean_tmp = sess.run(model_variables[index])
                    bn_variance_tmp = sess.run(model_variables[index + 1])
                    bn_gamma_tmp = sess.run(model_variables[index + 2])
                    bn_beta_tmp = sess.run(model_variables[index + 3])

                    bn_variance_tmp = np.sqrt(bn_variance_tmp.reshape([-1]) + GLOBAL_EPS)
                    bn_mean_tmp = bn_mean_tmp.reshape([-1])
                    bn_gamma_tmp = bn_gamma_tmp.reshape([-1])
                    bn_beta_tmp = bn_beta_tmp.reshape([-1])

                    index += 4
                    fmap_num  = conv_w_tmp.shape[0]
                    for f_index in range(fmap_num):
                        conv_w_tmp[f_index,:,:,:] = conv_w_tmp[f_index,:,:,:] * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index]
                        conv_b_tmp[f_index] = (conv_b_tmp[f_index] - bn_mean_tmp[f_index]) * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index] + bn_beta_tmp[f_index]

                    weight_list.append(conv_w_tmp.reshape([-1]))
                    bias_list.append(conv_b_tmp)
                else:
                    weight_list.append(conv_w_tmp.reshape([-1]))
                    bias_list.append(conv_b_tmp)

            if index < len(model_variables) and 'FC' in model_variables[index].name:
                # print(model_variables[index].name)
                # print(model_variables[index + 1].name)

                w_tmp = sess.run(model_variables[index])
                b_tmp = sess.run(model_variables[index + 1])

                w_tmp = w_tmp.transpose(1, 0)
                weight_list.append(w_tmp.reshape([-1]))
                bias_list.append(b_tmp.reshape([-1]))

                index += 2


        return layer_list, weight_list, bias_list

    def create_xnetlist_v3(self, sess):
        new_dict = {}

        for tmp in range(0, len(self.layer_dict)):
            cur_layer = self.layer_dict[str(tmp)].copy()

            if str(tmp) == '0':
                new_dict[str(tmp)] = cur_layer
                continue

            input_index = cur_layer['input_index']
            cur_index =  cur_layer['index']
            reduce_index = cur_layer['reduce_index']
            new_cur_index = cur_index - reduce_index

            if type(input_index) is list:
                new_input_index = []
                for index in input_index:
                    input_layer = self.layer_dict[str(index)]
                    input_layer_index = input_layer['index']
                    input_layer_reduce_index = input_layer['reduce_index']
                    new_input_layer_index = input_layer_index - input_layer_reduce_index
                    new_input_index.append(new_input_layer_index)
            else:
                input_layer = self.layer_dict[str(input_index)]
                input_layer_index = input_layer['index']
                input_layer_reduce_index = input_layer['reduce_index']
                new_input_index = input_layer_index - input_layer_reduce_index


            if new_cur_index == 4:
                a = 0
            cur_layer['input_index'] = new_input_index
            cur_layer['index'] = new_cur_index
            cur_layer['reduce_index'] = 0


            if cur_layer['layer_type'] == 'LAYER_CONVOLUTIONAL':
                cur_layer['name'] = 'conv_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_SCALE':
                cur_layer['name'] = 'scale_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_CONCAT':
                cur_layer['name'] = 'concat_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_NN_RESIZE':
                cur_layer['name'] = 'resize_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_MAX_POOLING':
                cur_layer['name'] = 'pool_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_AVE_POOLING':
                cur_layer['name'] = 'pool_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_ELTWISE_SUM':
                cur_layer['name'] = 'sum_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer
            elif cur_layer['layer_type'] == 'LAYER_FULL_CONNECTION':
                cur_layer['name'] = 'fc_' + str(new_cur_index)
                new_dict[str(new_cur_index)] = cur_layer


        weight_list = []
        bias_list = []

        # model_variable = sess._graph._collections['xnet_variable']
        model_variables = tf.get_collection('xnet_varibale')
        model_variables = [var for var in model_variables if self.model_name in var.name]

        index = 0
        while index < len(model_variables):
            name = model_variables[index].name


            # layer_index
            if 'Conv' in model_variables[index].name:
                print(model_variables[index].name)
                print(model_variables[index + 1].name)
                conv_w_tmp = sess.run(model_variables[index])
                conv_b_tmp = sess.run(model_variables[index + 1])
                conv_w_tmp = conv_w_tmp.transpose(3, 2, 0, 1)
                conv_b_tmp = conv_b_tmp.reshape([-1])

                index += 2
                # v_tmp =  model_variables[index].name
                if index < len(model_variables) and model_variables[index].name.find('BatchNorm')>0:
                    # print(model_variables[index].name)
                    # print(model_variables[index+1].name)
                    # print(model_variables[index+2].name)
                    # print(model_variables[index+3].name)

                    bn_mean_tmp = sess.run(model_variables[index])
                    bn_variance_tmp = sess.run(model_variables[index + 1])
                    bn_gamma_tmp = sess.run(model_variables[index + 2])
                    bn_beta_tmp = sess.run(model_variables[index + 3])

                    bn_variance_tmp = np.sqrt(bn_variance_tmp.reshape([-1]) + GLOBAL_EPS)
                    bn_mean_tmp = bn_mean_tmp.reshape([-1])
                    bn_gamma_tmp = bn_gamma_tmp.reshape([-1])
                    bn_beta_tmp = bn_beta_tmp.reshape([-1])

                    index += 4
                    fmap_num  = conv_w_tmp.shape[0]
                    for f_index in range(fmap_num):
                        conv_w_tmp[f_index,:,:,:] = conv_w_tmp[f_index,:,:,:] * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index]
                        conv_b_tmp[f_index] = (conv_b_tmp[f_index] - bn_mean_tmp[f_index]) * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index] + bn_beta_tmp[f_index]

                    weight_list.append(conv_w_tmp.reshape([-1]))
                    bias_list.append(conv_b_tmp)
                else:
                    weight_list.append(conv_w_tmp.reshape([-1]))
                    bias_list.append(conv_b_tmp)

            elif 'FC' in model_variables[index].name:
                # print(model_variables[index].name)
                # print(model_variables[index + 1].name)

                w_tmp = sess.run(model_variables[index])
                b_tmp = sess.run(model_variables[index + 1])

                w_tmp = w_tmp.transpose(1, 0)
                weight_list.append(w_tmp.reshape([-1]))
                bias_list.append(b_tmp.reshape([-1]))

                index += 2

            else:
                index += 1


        return new_dict, weight_list, bias_list

    def create_prototxt_and_caffemodel(self, sess, prototxt, model, is_caffe_model = False):
        # layer_list, weight_list, bias_list = self.create_xnetlist_v2(sess)
        layer_dict, weight_list, bias_list = self.create_xnetlist_v3(sess)

        # info_list = XnetLayerList2InfoList(layer_list)
        info_list = XnetLayerDict2InfoList(layer_dict)

        # #add final scale
        # info = {}
        # info['op'] = 'Scale'
        # info['name'] = 'final_scale'
        # info['bottom'] = [info_list[len(info_list)-1]['name']]
        # info['top'] = 'final_scale'
        # info_list.append(info)

        with open(prototxt, "w") as prototxt_file:
            data(prototxt_file)
            for info in info_list:
                write_node(prototxt_file, info)

        if is_caffe_model:
            import sys
            sys.path.insert(0, '/data6/shentao/Projects/Glass_remove_caffe/caffe-master/python')
            import caffe

            caffe.set_mode_cpu()

            net = caffe.Net(prototxt, caffe.TEST)

            count = 0
            for info in info_list:
                name = info['name']
                op = info['op']
                if op == 'Convolution':
                    net.params[name][0].data.flat = weight_list[count].flat
                    net.params[name][1].data.flat = bias_list[count].flat
                    count += 1
                if op == 'FullyConnected':
                    net.params[name][0].data.flat = weight_list[count].flat
                    net.params[name][1].data.flat = bias_list[count].flat
                    count += 1
                # if name == 'final_scale':
                #     net.params[name][0].data.flat = np.asarray([127.5]).flat #(x+1)/2 * 255-> x * 127.5 + 127.5
                #     net.params[name][1].data.flat = np.asarray([127.5]).flat

            # ------------------------------------------
            # Finish
            net.save(model)
            print("\n- Finished.\n")

    def create_xnetfile(self,sess):
        xnetname = self.model_name
        systime = time.strftime('%Y-%m-%d,%H:%M', time.localtime(time.time()))
        params_txt = 'xnet_' + xnetname + '.h'
        pf = open(params_txt, 'w')

        xnetversion = 'test'
        pf.write('/**xnetlib information*******\n')
        pf.write('*model generate time:%s\n' % systime)
        pf.write('*lib name           :%s\n' % xnetname)
        pf.write('*version num        :%s\n' % xnetversion)
        pf.write('****************/\n')

        ##pf.write('#include "xbase.h"\n')
        ##pf.write('#include "xnet_model.h"\n')

        pf.write('namespace %s {\n' % xnetname)
        pf.write('const ModelLayer model_layers[] = {\n')

        layer_list, weight_list, bias_list = self.create_xnetlist(sess)

        for layer in layer_list:
            param_str = ', '.join(layer)
            pf.write('{%s},\n' % param_str)

        pf.write('};\n')
        pf.write('const int nModelLayer = %d;\n' % len(layer_list))

        ## width,height,channels
        pf.write('const int img_width = %d;\n' % 128)
        pf.write('const int img_height = %d;\n' % 128)
        pf.write('const int input_dims = %d;\n' % self.input_dim)
        pf.write('const int cnn_type = %s;\n' % 'CLASSIFY')

        ## weights
        pf.write('const float weight[] = {\n')
        write_weights(pf, weight_list, 100)
        pf.write('\n};\n')

        ## bias
        pf.write('const float bias[] = {\n')
        write_weights(pf, bias_list, 30)
        pf.write('\n};\n')

        ## data mean
        data_mean = '128,128,128'
        data_std = '1,1,1'

        pf.write('const float data_mean[] = {%s};\n' % data_mean)
        pf.write('const float data_std[] = {%s};\n' % data_std)
        pf.write('}\n')
        pf.close






