import sys
sys.path.append('/home/gaofuxun/caffe/python')
import numpy as np
import caffe
from scipy import stats
import cv2
import os

caffe_root = '/home/gaofuxun/caffe/distribute/'
sys.path.insert(0, caffe_root + 'python')

caffe.set_device(1)
caffe.set_mode_gpu()


ft = 'fc8'  # The output of network
MODEL_FILE = '/data2/gaofuxun/liveness/RankIQA-master/src/FT/tid2013/deploy_vgg.prototxt'
PRETRAINED_FILE = '/data2/gaofuxun/liveness/RankIQA-master/models/ft_rank_tid2013/' +'my_siamese_iter_10000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)


#filename = [line.rstrip('\n') for line in open(test_file)]
file_dir = '/data2/gaofuxun/data/RankIQA/iqiyi_tid2013_train/'
savepath = '/data2/gaofuxun/data/RankIQA/iqiyi_tid2013_128_ten_crop_train/'

filename = []
for image_name in os.listdir(file_dir):
    filename.append(file_dir+image_name)

if not os.path.exists(savepath):
    os.makedirs(savepath)

roidb = filename
Num_Image = len(roidb)

def image_TT(img):
    im = cv2.resize(img, (168, 168))

    im_upper_left_ = im[0:128, 0:128]
    # cv2.imwrite('ul.jpg', im_upper_left_)
    im_upper_left = np.asarray(im_upper_left_)
    im_upper_left = im_upper_left[:, :, :].transpose([2,0,1])

    im_upper_right_ = im[0:128, 40:]
    # cv2.imwrite('ur.jpg', im_upper_right_)
    im_upper_right = np.asarray(im_upper_right_)
    im_upper_right = im_upper_right[:, :, :].transpose([2,0,1])

    im_lower_left_ = im[40:168, 0:128]
    # cv2.imwrite('ll.jpg', im_lower_left_) 
    im_lower_left = np.asarray(im_lower_left_)
    im_lower_left = im_lower_left[:, :, :].transpose([2,0,1])

    im_lower_right_ = im[40:168, 40:168]
    # cv2.imwrite('lr.jpg', im_lower_right_) 
    im_lower_right = np.asarray(im_lower_right_)
    im_lower_right = im_lower_right[:, :, :].transpose([2,0,1])

    im_central_ = im[20:148, 20:148]
    # cv2.imwrite('cen.jpg', im_central_) 
    im_central = np.asarray(im_central_)
    im_central = im_central[:, :, :].transpose([2,0,1])

    im_ul_flip = cv2.flip(im_upper_left_, -1)
    # cv2.imwrite('ulf.jpg', im_ul_flip) 
    im_ul_flip = np.asarray(im_ul_flip)
    im_ul_flip = im_ul_flip[:, :, :].transpose([2,0,1])

    im_ur_flip = cv2.flip(im_upper_right_, -1)
    im_ur_flip = np.asarray(im_ur_flip)
    im_ur_flip = im_ur_flip[:, :, :].transpose([2,0,1])

    im_ll_flip = cv2.flip(im_lower_left_, -1)
    im_ll_flip = np.asarray(im_ll_flip)
    im_ll_flip = im_ll_flip[:, :, :].transpose([2,0,1])

    im_lr_flip = cv2.flip(im_lower_right_, -1)
    im_lr_flip = np.asarray(im_lr_flip)
    im_lr_flip = im_lr_flip[:, :, :].transpose([2,0,1])

    im_c_flip = cv2.flip(im_central_, -1)
    im_c_flip = np.asarray(im_c_flip)
    im_c_flip = im_c_flip[:, :, :].transpose([2,0,1])

    return im_upper_left, im_upper_right, im_lower_left,\
           im_lower_right, im_central, im_ul_flip, \
           im_ur_flip, im_ll_flip, im_lr_flip, im_c_flip

for i in range(Num_Image):
    directory = roidb[i]
    print directory
    img = cv2.imread(directory)
    im_ul, im_ur, im_ll, im_lr, im_c, im_ulf, im_urf, im_llf,\
    im_lrf, im_cf = image_TT(img)
    
    out_ul = net.forward_all(data=np.asarray([im_ul]))
    out_ur = net.forward_all(data=np.asarray([im_ur]))
    out_ll = net.forward_all(data=np.asarray([im_ll]))
    out_lr = net.forward_all(data=np.asarray([im_lr]))
    out_c = net.forward_all(data=np.asarray([im_c]))
    out_ulf = net.forward_all(data=np.asarray([im_ulf]))
    out_urf = net.forward_all(data=np.asarray([im_urf]))
    out_llf = net.forward_all(data=np.asarray([im_llf]))
    out_lrf = net.forward_all(data=np.asarray([im_lrf]))
    out_cf = net.forward_all(data=np.asarray([im_cf]))

    total_score = out_ul[ft][0]+out_ur[ft][0]+out_ll[ft][0]+\
            out_lr[ft][0]+out_c[ft][0]+out_ulf[ft][0]+\
            out_urf[ft][0]+out_llf[ft][0]+out_lrf[ft][0]+\
            out_cf[ft][0]
    pre_score = total_score / 10.0
    #print out_ul[ft][0], out_ur[ft][0], out_ll[ft][0],\
    #      out_lr[ft][0], out_c[ft][0], out_ulf[ft][0],\
    #      out_urf[ft][0], out_llf[ft][0], out_lrf[ft][0],\
    #      out_cf[ft][0], pre_score
    cv2.imwrite(savepath+str(pre_score)[2:-1]+'.jpg', img)





"""
Num_Patch = 30
Num_Image = len(roidb)
feat = np.zeros([Num_Image,Num_Patch])
pre = np.zeros(Num_Image)
med = np.zeros(Num_Image)

for i in range(Num_Image):
    directory = roidb[i]
    print directory
    img = cv2.imread(directory)
    im = cv2.resize(img, (128, 128))
    im = np.asarray(im)
    for j in range(Num_Patch):   
        temp = im[:, :, :].transpose([2,0,1])

        out = net.forward_all(data=np.asarray([temp]))

        feat[i,j] = out[ft][0]
        pre[i] += out[ft][0]
    pre[i] /= Num_Patch
    med [i] = np.median(feat[i,:])
    print('pre: {}, med: {}'.format(pre[i], med[i]))
    cv2.imwrite(savepath+str(pre[i])+'.jpg', img)
"""
