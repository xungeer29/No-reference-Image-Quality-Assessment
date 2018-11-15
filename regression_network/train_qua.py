# -*- coding:utf-8 -*-

import math
import tensorflow as tf
# from data_op import *
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import inspect
import tensorlayer as tl
from Xtensorflow import *
import numpy as np
from utils import *
from math import cos,sin
import cv2

lr = 0.001
weight_decay = 0.0005
init_epoch = 50
epoch = 200

load_name = 'FaceAngle'
name = 'FaceAngle'
delta = 0.001

model_size = 48

class Face_Angle():
    def __init__(self, imgsize=128, batchsize=64, weight_decay = 0.0002, train = True, load_model_dir = 'load' ,save_model_dir = 'save' ):
        self.weight_decay = tf.constant(weight_decay, dtype=tf.float32)

        self.load_model_dir = load_model_dir
        self.save_model_dir = save_model_dir

        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = imgsize
        self.output_size = imgsize

        self.weight_decay =weight_decay
        self.input_colors = 1
        self.classnum = 1
        # self.classnum = 4 # 要拟合的label个数********************************

        ###========================== train graph ================================###
        self.input_images = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size, self.input_colors])

        self.regression_label = tf.placeholder(tf.float32, [self.batch_size, 1]) # **************************
        # self.regression_label = tf.placeholder(tf.float32, [self.batch_size, 4]) # label的placeholder占位
        self.regression_logits ,_ = self.network(self.input_images,
                                     self.batch_size,
                                     is_training = True,
                                     name = 'xnet',
                                     reuse=False,
                                     dropout=0.5)

        ###========================== test graph ================================###
        self.test_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size,self.input_colors])
        self.test_label = tf.placeholder(tf.float32, [self.batch_size, 1])
        self.logits_test_images, self.xnet = self.network(self.test_images,
                                                          self.batch_size,
                                                          is_training = False,
                                                          name = 'xnet',
                                                          reuse=True)

        ###========================== DEFINE TRAIN OPS ==========================###
        def MSE_loss(output,target):
            return tf.reduce_mean(tf.squared_difference(output, target))

        self.regression_loss = MSE_loss(self.regression_logits , self.regression_label) # *****************

        train_vars = tl.layers.get_variables_with_name("xnet", True, True)
        xnet_variables = tf.get_collection('xnet_varibale')
        self.g_vars = [var for var in xnet_variables if 'xnet'  in var.name]

        self.regression_optim = tf.train.AdamOptimizer(lr).minimize(self.regression_loss, var_list=train_vars)
        self.loadmodel()


    def network(self, inputs, batch_size,is_training, name,reuse = False, dropout = 1.0 ):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            xnet = Xtensorflow(inputs, is_train= is_training,weight_decay=self.weight_decay, model_name=name)

            conv1 = xnet.conv_with_bn_layer(0, [batch_size, 24, 24, 32], 5, 2, tf.nn.relu, index= 1)
            conv2 = xnet.conv_with_bn_layer(conv1, [batch_size, 12, 12, 64], 3, 2, tf.nn.relu)
            conv3 = xnet.conv_with_bn_layer(conv2, [batch_size, 6, 6, 64], 3, 2, tf.nn.relu)
            conv4 = xnet.conv_with_bn_layer(conv3, [batch_size, 3, 3, 128], 3, 2, tf.nn.relu)
            fc6 = xnet.fc_layer(conv4, [batch_size, 128], None, dropout=dropout)
            fn7 = xnet.bn_layer(fc6, activation=tf.nn.relu)
            fc8 = xnet.fc_layer(fn7, [batch_size, self.classnum], None)

            return xnet.get_network_output(), xnet


    def loadmodel(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        xnet_variables = tf.get_collection('xnet_varibale')
        self.sess.run(tf.variables_initializer(xnet_variables))
        self.saver = tf.train.Saver(self.g_vars,max_to_keep = 1)
        self.load("./checkpoint")

    def save(self, checkpoint_dir, step):
        generator_checkpoint_dir = os.path.join(checkpoint_dir, self.save_model_dir) + '/generator'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if not os.path.exists(generator_checkpoint_dir):
            os.makedirs(generator_checkpoint_dir)

        self.saver.save(self.sess, os.path.join(generator_checkpoint_dir, 'model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        if self.load_model_dir != None:
            checkpoint_dir = os.path.join(checkpoint_dir, self.load_model_dir)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir + '/generator')
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir + '/generator', ckpt_name))
                print ('load generator')

    def train_listfile(self, regression_iter, epoch_index):
        # print(len(self.regression_list))
        for i in range(int(len(self.regression_list) / self.batch_size)-1):

            if i % 200 == 0 and i >0:
                self.save("./checkpoint", epoch_index * 100000)


            live_batch, label = next(regression_iter)

            # chech whether the input is correct
            # print('label:{}'.format(label))
            # print('live_batch:{}'.format(live_batch))

            
            # start training
            live_batch = np.asarray(live_batch).astype(np.float32)
            live_batch = live_batch.reshape([self.batch_size,self.image_size,self.image_size,self.input_colors])
            
            batch = np.concatenate([live_batch])
            batch = np.array(batch) - 128.0
            
            label = label.reshape([self.batch_size, 1]) # 回归的label个数 ############*********************
            regression_logits, regression_loss, _ = self.sess.run([self.regression_logits, self.regression_loss, self.regression_optim],
                                            feed_dict={self.input_images: batch, self.regression_label: label})
            
            # print(name + " %d: [%d / %d] lr %f classify_loss %f regression_loss %f acc %f" % (epoch_index, i, (self.len / self.batch_size), lr, classify_loss, regression_loss, acc))
            print(name + '%d: [%d / %d] lr %f regression_loss %f' % (epoch_index, i, (len(self.regression_list) / self.batch_size), lr, regression_loss))
            # print regression_logits, regression.shape
            
    def train(self):
        self.regression_list, _ = self.read_quality_list(r'/data2/gaofuxun/data/RankIQA/iqiyi_tid2013_128_ten_crop_train.txt')
        random.shuffle(self.regression_list)

        from Xtensorflow.dataflow.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        from imgaug import augmenters as iaa
        import imgaug as ia

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            # iaa.Fliplr(0.5),
            sometimes(iaa.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.6, 0.95),
            #    rotate=(-10, 10),
                cval=128,
                mode=ia.ALL))
        ])


        def read_regression_image(sample):
            path = sample[0]
            # confidence_0 = sample[1] # yaw
            # confidence_1 = sample[2] # pitch
            # confidence_2 = sample[3] # roll
            confidence_0 = sample[1] # quality_score

            con = [confidence_0]
            # con = [confidence_0, confidence_1, confidence_2, confidence_3]

            image = cv2.imread(path, 0)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.reshape([1, self.image_size, self.image_size, 1])
            images_aug = seq.augment_images(image) # 数据增强
            # images_aug = image
            images_aug = images_aug.reshape([self.image_size, self.image_size])
            return images_aug, np.asarray(con)

        def GetImageListIter(image_list, batch_size, read_image_func, threads):
            dp = DataFromList(image_list)
            dp = MultiProcessMapDataZMQ(dp, threads, read_image_func, buffer_size=batch_size * 10)
            dp = BatchData(dp, batch_size)
            dp.reset_state()
            data_iter = dp.get_data()
            return data_iter

        self.regression_iter = GetImageListIter(self.regression_list, self.batch_size, read_regression_image, 16)
        ###============================= Training ===============================###
        for e in range(init_epoch + 1):
            self.train_listfile(self.regression_iter, e)

            if e % 1 == 0:
                self.save("./checkpoint", e * 100000)

    def cvimg_test(self, cv_mat):
        input_images = cv_mat.reshape(1,model_size,model_size,1)
        logits =  self.sess.run(self.logits_test_images, feed_dict={self.test_images: input_images})
        index = np.argmax(np.asarray(logits).reshape([-1])[0:2])
        return index, np.asarray(logits).reshape([-1])

    def read_quality_list(self, txt_path):
        f = open(txt_path, 'r')
        lines = f.readlines()

        samples = []
        for line in lines:
            line = line.strip()
            image_path = line

            image_path, yaw, pitch, roll, qua = line.split(' ')
            yaw = float(yaw)
            pitch = float(pitch)
            roll = float(roll) # aligned
            qua = float(qua)*1.85 # 将质量拓展到0-10范围

            samples.append([image_path, qua]) # ****************
            # samples.append([image_path, yaw, pitch, roll, qua]) # #########################3
            # print image_quality

        return samples, len(samples)

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) *sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

if __name__ == '__main__':
    # release comment to train
    gr = Face_Angle(imgsize=model_size , batchsize=512, load_model_dir= load_name , save_model_dir=name)
    gr.train()

    # test
    """
    gr = Face_Angle(imgsize=model_size, batchsize=1, load_model_dir=load_name, save_model_dir=name)
    test_image_path = '/data2/gaofuxun/data/RankIQA/iqiyi_aligned_face_val/'
    # f = open('./test_result.txt', 'w')

    num = 1
    total_num = len(os.listdir(test_image_path))
    for root, dirs, files in os.walk(test_image_path):
        for name in files:
            if name.endswith('.jpg'):
                test_image_path = os.path.join(root, name)
                img_row = cv2.imread(test_image_path)
                img_test = cv2.imread(test_image_path, 0)
                x, y = img_test.shape
                img_test = np.asarray(cv2.resize(img_test, (48, 48))).astype(np.float32) - 128
                conf, logits = gr.cvimg_test(np.asarray(img_test))
                draw_axis(img_row, logits[0], logits[1], 0, tdx = x/2, tdy = y/2, size = y/2)
                cv2.imwrite('/data2/gaofuxun/data/RankIQA/iqiyi_tid2013_128_ten_crop_test_pose_qua_no_argu/'+str(logits[3])+'.jpg', img_row)
                print('Finish {}/{}'.format(num, total_num))
                num += 1
    """
