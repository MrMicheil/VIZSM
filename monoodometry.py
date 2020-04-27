import numpy as np
import cv2
import os

class MonoOdometry(object):
    def __init__(self, img_file_path, pose_file_path, focal_length, pp, lk_params, detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0
    
        self.zmenaSnimku()

    def dalsiSnimok(self):
         return self.id < len(os.listdir(self.file_path)) 
        #    bool -- Boolean value denoting whether there are still frames in the folder to process

    def zmenaSnimku(self):

        if self.id < 2:
            self.old_frame = cv2.imread(self.file_path +str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(self.file_path + str(1).zfill(6)+'.png', 0)
            self.odometria()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.odometria()
            self.id += 1

    def detect(self, img):   

        p0 = self.detector.detect(img) 
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)
        #    np.array -- A sequence of points in (x, y) coordinate format denoting location of detected keypoint


    def vyrataj_suradnice(self):

        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten() 
        # We multiply by the diagonal matrix to fix our vector onto same coordinate axis as true values
    
    def vyrataj_realne_suradnice(self):
        
        riadok = self.riadok[self.id - 1].strip().split()
        x = float(riadok[3])
        y = float(riadok[7])
        z = float(riadok[11])

        realne_suradnice = np.array([[x], [y], [z]])
        return realne_suradnice.flatten()


    def odometria(self):

        if self.n_features < 2000:                          #
            self.p0 = self.detect(self.old_frame)

        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        
        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]


        if self.id < 2:
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R, self.t, self.focal, self.pp, None)
        else:
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), self.focal, self.pp, None)

            absolute_scale = self.get_absolute_scale()              #read document
            if (absolute_scale > 0.1):                              #Ak som nemal dataset dal som tam len absolute_scale = 0.2 Je to udaj napr. o rychlosti kamery
                self.t = self.t + absolute_scale * self.R.dot(t)    
                self.R = R.dot(self.R)

        self.n_features = self.good_new.shape[0]
        # Save good points

##########      ONLY WITH DATASET      #############

    def get_absolute_scale(self):

        riadok = self.riadok[self.id - 1].strip().split()
        x_prev = float(riadok[3])
        y_prev = float(riadok[7])
        z_prev = float(riadok[11])
        riadok = self.riadok[self.id].strip().split()
        x = float(riadok[3])
        y = float(riadok[7])
        z = float(riadok[11])

        true_vect = np.array([[x], [y], [z]])
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        
    #   alternative 
    #   return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))    
        return np.linalg.norm(true_vect - prev_vect)

