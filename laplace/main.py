import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import edge_detect as ed
from scipy import signal


infile = 'resources/lena.png'

im = Image.open(infile)
arr = np.array(im)
gray = ed.to_gray(arr)
gray = ed.averageFilter(gray, 5)
gray = ed.median_filter(gray, 3)
gray = ed.median_filter(gray, 3)
#gray = ed.median_filter(gray, 3)
lap = ed.laplacian(gray)
norm = ed.set_min_max(lap)
(zc, stdev) = ed.find_zero_cross(lap)
vals = np.sort(stdev, axis=None)
vals = [val for val in vals if val != 0]

ind = int(3*len(vals)/4)
th = vals[ind]
print(f'TH: {th}')
for i in range(0, zc.shape[0]):
    for j in range(0, zc.shape[1]):
        if(stdev[i, j] > th):
            zc[i, j] = 255

ed.print_to_file(gray, 'output/gray.png')
ed.print_to_file(norm, 'output/norm.png')
ed.print_to_file(zc, 'output/edges.png')