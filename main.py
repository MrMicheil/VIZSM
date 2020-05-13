import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monoodometry import MonoOdometry
import os

img_path = 'C:\\data_odometry_gray\\dataset\\sequences\\25\\image_0\\'
pose_path = 'C:\\data_odometry_poses\\dataset\\poses\\25.txt'

# focal V5, V7, V9, V11, V13 = 187.5      V6, V8, V10 = 375

#                                                                                       35/10/5   1200      32       375   
# example: image resolution of 640x480, we compute the OpenCV focal length as follows: 35mm * (640 pixels / 32 mm) = 700 pixels
# for blender is optimal between 350 and 3500 than for a setting of 35

# pp - principal point
# double cx = (newImgSize.width)*0.5;
# double cy = (newImgSize.height)*0.5; 

#                          Video ulica (25)            Video s metrom (27)                    
SenzorSize = 32            #32                         9.5
focalInMM = 10             #10                         27
width = 1200               #1200                       1024
height = 380               #380                        768
focal = focalInMM*(width/SenzorSize)
pp = (width*0.5, height*0.5)
#Kitty focal = 718.8560      pp = (607.1928, 185.2157)        dataset 00

# Definovanie parametrov pre  lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

Odometry = MonoOdometry(img_path, pose_path, focal, pp, lk_params) 
Cesta = np.zeros(shape=(800, 800, 3))

while(Odometry.dalsiSnimok()):

    Snimka = Odometry.current_frame
    cv.imshow('Zaznam', Snimka)
    k = cv.waitKey(1)
    if k == 27:
        break

    Odometry.zmenaSnimku()

    VyratSur = Odometry.vyrataj_suradnice()
    RealneSur = Odometry.vyrataj_realne_suradnice()

    #Vypis do terminalu
    print("x: {}, z: {}, y: {}".format(*[str(pt) for pt in VyratSur]))
    #Vypis na obrazovku
    x, y, z = [int((x)) for x in VyratSur]
    Cesta = cv.circle(Cesta, (x * 3 + 400, z * 3 + 100), 1, list((0, 255, 0)), 4)
     
    x, y, z = [int(round(x)) for x in RealneSur]

    Cesta = cv.circle(Cesta, (x * 3 + 400, z * 3 + 500), 1, list((0, 0, 255)), 4)
    cv.putText(Cesta, 'Vizualna odometria:', (100, 100), cv.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)
    cv.putText(Cesta, 'Skutocna pozicia:', (100, 500), cv.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)
    cv.imshow('Trajektoria', Cesta)

