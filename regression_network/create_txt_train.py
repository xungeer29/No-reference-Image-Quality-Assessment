# -*- coding:utf-8 -*-

import os
import cv2

image_path_train = '/data2/gaofuxun/data/RankIQA/iqiyi_tid2013_train/'
train_txt = open('/data2/gaofuxun/data/RankIQA/iqiyi_tid2013_train.txt', 'w')

for root, dirs, files in os.walk(image_path_train):
    for name in files:
        if name.endswith('.jpg'):
            train_txt.write(image_path_train+name+'\n')

train_txt.close()
            
