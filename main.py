import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monovideoodometery import MonoVideoOdometery
import os



img_path = 'C:\\data_odometry_gray\\dataset\\sequences\\00\\image_0\\'
pose_path = 'C:\\data_odometry_poses\\dataset\\poses\\00.txt'


# focal = 718.8560      pp = (607.1928, 185.2157)
# focal V5, V7, V9, V11, V13 = 187.5      V6, V8, V10 = 375

#                                                                                       35/10/5   1200      32       375   
# example: image resolution of 640x480, we compute the OpenCV focal length as follows: 35mm * (640 pixels / 32 mm) = 700 pixels
# for blender is optimal between 350 and 3500 than for a setting of 35 (we have 35,10 and 5 :D)

# pp - principal point
# double cx = (newImgSize.width)*0.5;
# double cy = (newImgSize.height)*0.5; 

focal = 187.5
pp = (600, 190)


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

#wimSize  This determines the integration window size. Small windows are more sensitive to noise 
#         and may miss larger motions. Large windows will “survive” an occlusion.
#criteria criteria has two interesting parameters here - 
#         - the max number (10 above) of iterations and 
#         - epsilon (0.03 above). More iterations means a more exhaustive search, and a smaller 
#        epsilon finishes earlier. These are primarily useful in exchanging speed vs accuracy, but mainly stay the same.