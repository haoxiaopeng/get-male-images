import numpy as np
import sys
import caffe
from scipy import stats
import cv2
import os
import math
#caffe_root = '/home/xialei/caffe/distribute/'
#sys.path.insert(0, caffe_root + 'python')

caffe.set_device(1)
caffe.set_mode_gpu()

filenames=[]
for fl in os.listdir('./diandian_out/'):
    filenames.append(fl)
ft = 'BigResNet/FC2'  # The output of network
MODEL_FILE = './BigResNet_nobn.prototxt'
#PRETRAINED_FILE = './models/ft_live/' +'my_siamese_iter_50000.caffemodel'
PRETRAINED_FILE = './BigResNet_nobn.caffemodel'

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

    image_points = np.array([
                                (int(out[0][62]*y),int(out[0][149]*y)),     # Nose tip
                                (int(out[0][7]*y),int(out[0][94]*y)),     # Chin
                                (int(out[0][31]*y),int(out[0][118]*y)),     # Left eye left corner
                                (int(out[0][43]*y),int(out[0][130]*y)),     # Right eye right corne
                                (int(out[0][66]*y),int(out[0][153]*y)),     # Left Mouth corner
                                (int(out[0][72]*y),int(out[0][159]*y))      # Right mouth corner
                          ], dtype="double")
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])
    img_size=im.shape
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )

    #p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    #p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    #cv2.line(im, p1, p2, (255,0,0), 2)

    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    #q=quaternion()
    w=math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0]/theta
    y = math.sin(theta / 2)*rotation_vector[1][0]/theta
    z = math.sin(theta / 2)*rotation_vector[2][0]/theta
    # Display image
    alpha = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    angle = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    #angle = angle*180/math.pi
    #alpha = alpha*180/math.pi
    print "pitch \n:{0}".format(int((angle/math.pi)*180))
    print "yaw \n:{0}".format(int((yaw/math.pi)*180))
    print "alpha \n:{0}".format(int((alpha/math.pi)*180))
    x=draw.shape[0]
    y=draw.shape[1]
    cv2.circle(draw,(int(out[0][62]*y),int(out[0][149]*x)),2,(255,0,0),10) 
    cv2.circle(draw,(int(out[0][7]*y),int(out[0][94]*x)),2,(255,0,0),10) 
    cv2.circle(draw,(int(out[0][31]*y),int(out[0][118]*x)),2,(255,0,0),10) 
    cv2.circle(draw,(int(out[0][43]*y),int(out[0][130]*x)),2,(255,0,0),10) 
    cv2.circle(draw,(int(out[0][66]*y),int(out[0][153]*x)),2,(255,0,0),10) 
    cv2.circle(draw,(int(out[0][72]*y),int(out[0][159]*x)),2,(255,0,0),10) 
    
    cv2.imwrite("../"+i,draw)


    #os._exit(0)
