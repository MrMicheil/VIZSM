# VISUAL ODOMETRY 

Snažil som sa pomeniť niečo v kóde, keď si niečo napadne skús na to mrknúť očkom. Úplne dole nájdeš porovnanie kódov na screenshotoch. 
Neviem či si pozeral aj ty niečo ale ja som si prešiel všetky funkcie a tu ich zbežne napíšem. 
Niektoré sceny už mam ready len neviem ako ti to poslať. Ďalší postup zajtra. 

**main**
Bez poznámky
```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monoodometry import MonoOdometry
import os

img_path = 'C:\\data_odometry_gray\\dataset\\sequences\\22\\image_0\\'
pose_path = 'C:\\data_odometry_poses\\dataset\\poses\\22.txt'
```


S týmto som sa chvíľu pohral, hlavne keď som menil focal pri vyrenderovaný. 
Čím nižší focal v mm tým väčší záber na scénu a vie viac bodov zobrať => presnejšia odometria bez scalovania
Teóriu okolo "čo je to focal a pp" dám rovno do dokumentácie (ohnisko kamery a stred obrazu)
```python
# focal = 718.8560      pp = (607.1928, 185.2157)
# focal V5, V7, V9, V11, V13 = 187.5      V6, V8, V10 = 375

#                                                                                       35/10/5   1200      32       375   
# example: image resolution of 640x480, we compute the OpenCV focal length as follows: 35mm * (640 pixels / 32 mm) = 700 pixels
# for blender is optimal between 350 and 3500 than for a setting of 35 (we have 35,10 and 5 :D)

# pp - principal point
# double cx = (newImgSize.width)*0.5;
# double cy = (newImgSize.height)*0.5; 

SenzorSize = 32
focalInMM = 5
width = 1200
height = 380

focal = focalInMM*(width/SenzorSize)
pp = (width*0.5, height*0.5)
```

![Focal](https://cdn-7.nikon-cdn.com/Images/Learn-Explore/Photography-Techniques/2009/Focal-Length/Media/focal-length-graphic.jpg)

![enter image description here](https://i.stack.imgur.com/1mUcU.png)

#### Parameters for lucas kanade optical flow
*#wimSize  This determines the integration window size. Small windows are more sensitive to noise 
         and may miss larger motions. Large windows will “survive” an occlusion.
#criteria criteria has two interesting parameters here - 
         - the max number (10 above) of iterations and 
         - epsilon (0.03 above). More iterations means a more exhaustive search, and a smaller 
        epsilon finishes earlier. These are primarily useful in exchanging speed vs accuracy, but mainly stay the same.*
        
 Najlepšie vysvetlenie je asi toto : do dokumentácie hodím screenshoty nejakého porovnania z inými. 
```python
lk_params = dict( winSize  = (21,21),
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
```
Odometry - naša funkcia 
Cesta - vytvorenie poľa kam zapisujem súradnice ktoré sa zobrazia na mape
```python
Odometry = MonoOdometry(img_path, pose_path, focal, pp, lk_params)
Cesta = np.zeros(shape=(600, 800, 3))
```


Pozn. Mal som nápad že to prerobiť na for cyclus na počet snímok ale to by bolo už úplne hlúpe nie ? 
```python
while(Odometry.dalsiSnimok()):
    Snimka = Odometry.current_frame
    cv.imshow('Zaznam', Snimka)
    k = cv.waitKey(1)
    if k == 27:
        break
    Odometry.zmenaSnimku()
```
Funkciu Realne suradnice som trocha zmenil vid riad 115 screen 4
```python
    VyratSur = Odometry.vyrataj_suradnice()
    RealneSur = Odometry.vyrataj_realne_suradnice()
```
 Toto som ešte nemenil, až na koniec keď budeme vedieť čo všetko chceme vypísať (max som to poprehadzoval :D)
 Akurát som to vyscaloval  " *4" kvôli tomu že mi sa hýbeme v desiatkach metroch a oni v stovkách
```python
    print("Chyba: ", np.linalg.norm(VyratSur - RealneSur))
    print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in VyratSur]))
    print("Realne_x: {}, Realne_y: {}, Realne_z: {}".format(*[str(pt) for pt in RealneSur]))
 
    x, y, z = [int(round(x)) for x in VyratSur]
    Cesta = cv.circle(Cesta, (x * 4 + 400, z * 4 + 100), 1, list((0, 0, 255)), 4)
    
    x, y, z = [int(round(x)) for x in RealneSur]
    Cesta = cv.circle(Cesta, (x * 4 + 400, z * 4 + 100), 1, list((0, 255, 0)), 4)

    cv.imshow('Trajektoria', Cesta)
 ```


**monoodometry**

Názvy premenných môžem zmeniť aj funkcií čo som už aj začal. 

```python
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
```
 Tu som odstranil try expect, to by som určite ja neprogramoval :D mám to vrátiť ? 
```python
          with open(pose_file_path) as f:
            self.riadok = f.readlines()
        self.zmenaSnimku()
```
 Podmienka celého programu
```python
      def dalsiSnimok(self):
         return self.id < len(os.listdir(self.file_path)) 
        #    bool -- Boolean value denoting whether there are still frames in the folder to process
```
 Ak vieš o nejakom inom sposobe ako v pythone prechadzať snimkami vo folderi sem s ním :D 
 Ja zmením počet čísiel v názve z 6 na 4 
 
 Vytvoril som funkciu pre prvú snímku ktorú volám v podmienke, zmena oproti originálu. Pozri screen 3 
```python
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
```
Funkcia detector vracia akekolvek vyrazne body kde su výrazne prechody medzi pixelmi. Poznáme FAST, Good features to track, ORB, SIFT, SURF, F
metódu FAST sme si vybrali kvoli že je menej vypočtovo naročný ako napr. SIFT alebo SURF. 
```python
    def detect(self, img):   
        p0 = self.detector.detect(img) 
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)
        #    np.array -- A sequence of points in (x, y) coordinate format denoting location of detected keypoint
```
![Detector FAST](http://avisingh599.github.io/images/visodo/fast.png)


Funkcia na vyrátanie suradnic z odometrie 
Tu nie je čo meniť....
```python
    def vyrataj_suradnice(self):
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)
        return adj_coord.flatten() 
        # We multiply by the diagonal matrix to fix our vector onto same coordinate axis as true values

```
Funkcia na vyrátanie suradnic z datasetu
Zmenil som počet stlpcov v datasete 
A teraz sa to ráta priamo v tejto funkcii narozdiel od originálu kde sa ta funkcia iba odkazuje na funkciu absolute_scale ... teraz absolute_scale môžeme vypínať ak nemáme dataset (dataset by sme nemuseli mať)       Pozri screen 4
```python
    def vyrataj_realne_suradnice(self):      
        riadok = self.riadok[self.id - 1].strip().split()
        x = float(riadok[0])
        y = float(riadok[1])
        z = float(riadok[2])
        realne_suradnice = np.array([[x], [y], [z]])
        return realne_suradnice.flatten()
```
Tu je pôvodne jedna funkcia rozdelená na 2 ako som písal vyššie nech je rozdiel orpoti originálu Pozri screen 3
```python
    def odometriaFirst(self):
        self.p0 = self.detect(self.old_frame)
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1] 
        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R, self.t, self.focal, self.pp, None)
        self.n_features = self.good_new.shape[0] 
    def odometria(self):
        if self.n_features < 2000:                          #
            self.p0 = self.detect(self.old_frame)
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        print(self.p1)     
        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]
```
## Tretia kľúčová funkcia je pre Essential Matrix Estimation

####  parametre : 
self.good_new good_old focal pp poznáme 


Trocha externého kopirovania
##### Essential Matrix Estimation

Once we have point-correspondences, we have several techniques for the computation of an essential matrix. The essential matrix is defined as follows:  yT1Ey2=0y1TEy2=0  Here,  y1y1,  y2y2  are homogenous normalised image coordinates. While a simple algorithm requiring eight point correspondences exists  a more recent approach that is shown to give better results is the five point algorithm. It solves a number of non-linear equations, and requires the minimum number of points possible, since the Essential Matrix has only five degrees of freedom.

##### RANSAC

If all of our point correspondences were perfect, then we would have need only five feature correspondences between two successive frames to estimate motion accurately. However, the feature tracking algorithms are not perfect, and therefore we have several erroneous correspondence. A standard technique of handling outliers when doing model estimation is RANSAC. It is an iterative algorithm. At every iteration, it randomly samples five points from out set of correspondences, estimates the Essential Matrix, and then checks if the other points are inliers when using this essential matrix. The algorithm terminates after a fixed number of iterations, and the Essential matrix with which the maximum number of points agree, is used.

## Štvrtá funkcia ktorá sa nedá zmeniť R, t 
hľadá rotačnú maticu R a translačný vector t - čiže naša vzdialenosť na snímku
For every pair of images, we need to find the rotation matrix R and the translation vector t, which describes the motion of the vehicle between the two frames. The vector t can only be computed upto a scale factor in our monocular scheme.
```python
        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), self.focal, self.pp, None)
```
Ešte zopár linkov 
[findEssentialMat](https://kite.com/python/docs/cv2.findEssentialMat)
[cv2.recoverPose](https://kite.com/python/docs/cv2.recoverPose)


### absolute_scale 
Oficiálne na takéto niečo potrebuješ jeden údaj napr rýchlosť kamery alebo udaj o zrýchlení alebo čokoľvek. aby vedel či sa vlastne posunul o jeden mm alebo o jeden km .. teda v kacýh jendotkách ma scalovať prechody medzi snimkami... Asi som mal radšej pripojiť link 
[absolute scale](https://www.researchgate.net/post/Absolute_scale_estimation_for_monocular_visual_odometry)
```python
        absolute_scale = 0.2             
        if (absolute_scale > 0.1):                              #Ak som nemal dataset dal som tam len absolute_scale = 0.2 Je to udaj napr. o rychlosti kamery
            self.t = self.t + absolute_scale * self.R.dot(t)    
            self.R = R.dot(self.R)
        self.n_features = self.good_new.shape[0]    
```
Funkcia na ratanie absolute scale Funguje iba ak mame dataset .. inac som tam dal pernamentne scale 0.2 (kamera v blendery aktualne sa mi hýbe medzi snimkami 1 m za sekundu a mam 5 snímkov v sekudne :D a celkom slušne to funguje aj bez odometrie) 
Navrhujem vymeniť ten posledný riadok v tej funkcii ... je to matematicky to iste nie ? 
```python
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
    #   alternative 
    #   return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))    
        return np.linalg.norm(true_vect - prev_vect)
```
 
