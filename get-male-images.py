import shutil
import numpy as np
import mxnet as mx
import argparse
import cv2
import time
from core.symbol import P_Net, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector
import caffe
import os
import sys
from wide_resnet import WideResNet
def test_net(prefix, epoch, batch_size, ctx,
             thresh=[0.3, 0.3, 0.4], min_face_size=120,
             stride=2, slide_window=False):

    faceCount = 0

    detectors = [None, None, None] #face detect

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    #gender part
    model_str='./insightface/gender-age/model/model,0'
    _vec = model_str.split(',')
    prefix = _vec[0]
    epoch = int(_vec[1])
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)

    # wowoface pose; output have 9 value, 0正脸，1，2，3，4微侧脸，5，6，7，8大侧脸
    ft_face_pose = 'classifier'  # The output of network
    face_pose_prot = './face_keypoint/wowofaceposeandroid.prototxt'
    face_pose_model= './face_keypoint/wowofaceposeandroid.caffemodel'
    face_pose_net = caffe.Net(face_pose_prot, face_pose_model, caffe.TEST)

    #face key point
    ft_keypoint = 'BigResNet/FC2'  # The output of network
    keypoint_prot = './face_keypoint/BigResNet_nobn.prototxt'
    keypoint_model= './face_keypoint/BigResNet_nobn.caffemodel'
    keypoint_net = caffe.Net(keypoint_prot, keypoint_model, caffe.TEST)

    for imgFile in os.listdir('./'):
        if '.jpg' in imgFile:
            print imgFile
            img = cv2.imread('./' + imgFile)
            t1 = time.time()

            if img is None: continue
            boxes, boxes_c = mtcnn_detector.detect_pnet(img)
            if boxes_c is None:continue
            boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
            if boxes_c is None:continue
            boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)
            if boxes_c is None:continue

            print 'time: ',time.time() - t1
            font = cv2.FONT_HERSHEY_SIMPLEX
            faces = np.empty((1, 112, 112, 3))
            if boxes_c is not None:
                draw = img.copy()
                print boxes_c
                for b in boxes_c:
                    if b[4] < 0.98: continue
                    faceCount += 1
                    #cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 3)
                    #cv2.putText(draw, '%.3f'%b[4], (int(b[0]), int(b[1])), font, 1, (255, 255, 255), 2)
            #        cv2.imwrite('./diandian_out/' + imgFile, draw[int(b[1]):int(b[3]),int(b[0]):int(b[2])]);
                    if b[1]<0:
                        b[1]=0
                    if b[0]<0:
                        b[0]=0
                    xmin = int(b[0])
                    ymin = int(b[1])
                    xmax = int(b[2])
                    ymax = int(b[3])
                    w=img.shape[0]
                    h=img.shape[1]
                    wd = int((xmax-xmin)*0.15)
                    hd = int((ymax-ymin)*0.15)
                    xmin=max(xmin-wd,0)
                    ymin=max(ymin-hd,0)
                    xmax=min(xmax+wd,w)
                    ymax=min(ymax+hd,h)
                    #cv2.imwrite('./diandian_out/'+imgFile,draw[ymin:ymax,xmin:xmax])
                    # nimg is face image
                    nimg = cv2.resize(img[ymin:ymax,xmin:xmax, :], (112, 112))
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    aligned = np.transpose(nimg, (2,0,1))
                    input_blob = np.expand_dims(aligned, axis=0)
                    data = mx.nd.array(input_blob)
                    db = mx.io.DataBatch(data=(data,))
                    model.forward(db, is_train=False)
                    ret = model.get_outputs()[0].asnumpy()
                    g = ret[:,0:2].flatten()
                    gender = np.argmax(g)
                    a = ret[:,2:202].reshape( (100,2) )
                    a = np.argmax(a, axis=1)
                    age = int(sum(a))
                    
                    #gender: 0 is femal; 1 is male
                    if gender==0:
                        continue

   #                 print str(gender)+' '+str(age)
                    ##### face pose ##########
                    fimg = cv2.resize(img[ymin:ymax,xmin:xmax, :], (128, 128),interpolation=cv2.INTER_CUBIC)
                    fimg = np.asarray(fimg)
                    temp = fimg.transpose([2,0,1])
                    face_out = face_pose_net.forward(data=np.asarray([temp]))[ft_face_pose]
                    #if 大侧脸概率大于0.6,就被认为是不符合要求的脸 
                    if face_out[0][5]+face_out[0][6]+face_out[0][7]+face_out[0][8]>0.6:
                        continue
                    #else:
                    #   shutil.copy('./diandian_face/'+imgFile,'./diandian_fatface/'+imgFile)
                    
                    #########  key  point ########
                    keypoint_out = keypoint_net.forward(data=np.asarray([temp]))[ft_keypoint]
                    face_h = ymax-ymin
                    face_w = xmax-xmin
                    cv2.circle(draw,(int(keypoint_out[0][62]*face_w+xmin),int(keypoint_out[0][149]*face_h+ymin)),2,(255,0,0),10)
                    cv2.circle(draw,(int(keypoint_out[0][3]*face_w+xmin),int(keypoint_out[0][90]*face_h+ymin)),2,(255,0,0),10)
                    cv2.circle(draw,(int(keypoint_out[0][11]*face_w+xmin),int(keypoint_out[0][98]*face_h+ymin)),2,(255,0,0),10)
                    cv2.circle(draw,(int(keypoint_out[0][7]*face_w+xmin),int(keypoint_out[0][94]*face_h+ymin)),2,(255,0,0),10)
                    cv2.circle(draw,(int(keypoint_out[0][85]*face_w+xmin),int(keypoint_out[0][172]*face_h+ymin)),2,(255,0,0),10)
                    
                    cv2.imwrite('./male_face_imgs/'+imgFile, draw)
                # cv2.waitKey(0)
            #print 'Detected Face Num:', faceCount



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['model/pnet', 'model/rnet', 'model/onet'], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.5, 0.5, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=40, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = mx.gpu(args.gpu_id)
    if args.gpu_id == -1:
        print 'You are Using CPU Mode'
        ctx = mx.cpu(0)
    else:
        print 'GPU Model'
    test_net(args.prefix, args.epoch, args.batch_size,
             ctx, args.thresh, args.min_face,
             args.stride, args.slide_window)
