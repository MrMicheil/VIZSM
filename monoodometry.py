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
        with open(pose_file_path) as f:
            self.riadok = f.readlines()
        self.zmenaSnimku()
    #Funnkcia na vratenie ID snimku
    def dalsiSnimok(self):

         return self.id < len(os.listdir(self.file_path)) 

    #Funkcia na zmenu snimku 
    def zmenaSnimku(self):
        
        if self.id < 2:
            self.old_frame = cv2.imread(self.file_path +str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(self.file_path + str(1).zfill(6)+'.png', 0)
            self.odometriaFirst()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.odometria()
            self.id += 1

    #Funkcia na detegciu hrán
    def detect(self, img):   

        p0 = self.detector.detect(img) 
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    #Funkcia na vyratanie suradnic z visualnej odometrie
    def vyrataj_suradnice(self):
 
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)
        return adj_coord.flatten() 
 
    #Funkcia na vycitanie suradnic z txt suboru ak mame dataset s poziciou framu     
    def vyrataj_realne_suradnice(self):

        riadok = self.riadok[self.id - 1].strip().split()
        x = float(riadok[0])
        y = float(riadok[1])
        z = float(riadok[2])
        realne_suradnice = np.array([[x], [y], [z]])
        return realne_suradnice.flatten()
    
    #Funkcia na vyratanie odometrie pre prvu snimku 
    def odometriaFirst(self):

        self.p0 = self.detect(self.old_frame)
        #Funkcia Lucas Kanade pre zistenie totoznych hran medzi snimkami
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1] 
        #Funkcia na najdenie zakladnej matice
        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        #Funkcia na vyratanie R a t 
        _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R, self.t, self.focal, self.pp, None)
        self.n_features = self.good_new.shape[0] 
        
    #Funkcia na ratanie odometrie pre vsetky ostatne snimky
    def odometria(self):
        
        if self.n_features < 2000:                       
            self.p0 = self.detect(self.old_frame)

        #Funkcia Lucas Kanade pre zistenie totoznych hran medzi snimkami
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]
        #Funkcia na najdenie zakladnej matice
        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        #Funkcia na vyratanie R a t 
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), self.focal, self.pp, None)


        #absolute_scale = self.get_absolute_scale()     <<<<====== iba ak je dostupny dataset
        absolute_scale = self.get_absolute_scale()         #mierka pohybu kamery     
                                    #pre ulicu je to 0.4
                                    #pre video s metrom je to 2
        if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):  #Ak som nemal dataset dal som tam len absolute_scale = 0.2 Je to udaj napr. o rychlosti kamery
            #Rovnice na vyratanie polohy kamery
            self.t = self.t + absolute_scale * self.R.dot(t)     
            self.R = R.dot(self.R)

        self.n_features = self.good_new.shape[0]
        

##########      Iba ak je dostupny dataset      #############

    #Funkcia na vypocitanie mierky z datasetu (scalu)
    def get_absolute_scale(self):

        riadok = self.riadok[self.id - 1].strip().split()
        x_prev = float(riadok[0])  
        y_prev = float(riadok[1])   
        z_prev = float(riadok[2])  
        riadok = self.riadok[self.id].strip().split()
        x = float(riadok[0])   
        y = float(riadok[1])
        z = float(riadok[2])    

        true_vect = np.array([[x], [y], [z]])
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
          
        return np.linalg.norm(true_vect - prev_vect)

