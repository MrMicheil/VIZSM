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
    
        try:
            if not all([".png" in x for x in os.listdir(img_file_path)]):
                raise ValueError("img_file_path is not correct and does not exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.riadok = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.zmenaSnimku()

    def dalsiSnimok(self):
         return self.id < len(os.listdir(self.file_path)) 
        #    bool -- Boolean value denoting whether there are still frames in the folder to process
    
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


