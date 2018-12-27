import numpy as np
import sys
import caffe
from scipy import stats
import cv2
import os

#caffe_root = '/home/xialei/caffe/distribute/'
#sys.path.insert(0, caffe_root + 'python')

caffe.set_device(1)
caffe.set_mode_gpu()

filenames=[]
for fl in os.listdir('./diandian_out/'):
    filenames.append(fl)
ft = 'classifier'  # The output of network
MODEL_FILE = './wowofaceposeandroid.prototxt'
#PRETRAINED_FILE = './models/ft_live/' +'my_siamese_iter_50000.caffemodel'
PRETRAINED_FILE = './wowofaceposeandroid.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
for i in filenames:
    print i
    try:
        im=cv2.imread('./diandian_out/'+i)
    except:
        continue
    x= int(im.shape[0])
    y= int(im.shape[1])
    draw =im.copy()
    im=cv2.resize(im,(128,128),interpolation=cv2.INTER_CUBIC)
    im=np.asarray(im)
    temp=im.transpose([2,0,1])
    out = net.forward(data=np.asarray([temp]))[ft]
    print out
    #os._exit(0)
