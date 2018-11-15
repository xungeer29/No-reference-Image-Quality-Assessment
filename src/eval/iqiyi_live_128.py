import numpy as np
import sys
sys.path.append('/home/gaofuxun/caffe/python')
import caffe
from scipy import stats
import cv2
import os


caffe_root = '/data2/gaofuxun/caffe/distribute/'
sys.path.insert(0, caffe_root + 'python')

caffe.set_device(1)
caffe.set_mode_gpu()


ft = 'fc8'  # The output of network
MODEL_FILE = '/data2/gaofuxun/liveness/RankIQA-master/src/FT/live/deploy_vgg.prototxt'
PRETRAINED_FILE = '/data2/gaofuxun/liveness/RankIQA-master/models/ft_live/' +'my_siamese_iter_20000.caffemodel'
#tp = 'FT_all' 

net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

#res_dir = '/results/live/'

#srocc_file = open(  res_dir+ tp + '_srocc' + '.txt', "w")
#lcc_file = open( res_dir + tp + '_lcc' + '.txt', "w")


# test_file = './data/' + 'ft_live_test.txt'

# filename = [line.rstrip('\n') for line in open(test_file)]
file_dir = '/data2/gaofuxun/data/RankIQA/iqiyi_aligned_face_val/'
savepath = '/data2/gaofuxun/data/RankIQA/iqiyi_live_128/'
filename = []
for image_name in os.listdir(file_dir):
    filename.append(file_dir+image_name)

if not os.path.exists(savepath):
    os.makedirs(savepath)

roidb = filename

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
        # x =  im.shape[0]
        # y = im.shape[1]
        # x_p = np.random.randint(x-128,size=1)[0]
        # y_p = np.random.randint(y-128,size=1)[0] 
 
        temp = im[:, :, :].transpose([2,0,1])

        out = net.forward_all(data=np.asarray([temp]))

        feat[i,j] = out[ft][0]
        pre[i] += out[ft][0]
    pre[i] /= Num_Patch
    med [i] = np.median(feat[i,:])
    print('pre: {}, med: {}'.format(pre[i], med[i]))
    cv2.imwrite(savepath+str(pre[i])+'.jpg', img)



