import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import edge_detect.edge_detect as ed
from scipy import signal

# config
filt_ratio = 1/2
min_max_amp = 5


infile = 'resources/face.png'

rgb = ed.load_image(infile)
gray = ed.to_gray(rgb)
gray = ed.averageFilter(gray, 5)
gray = ed.median_filter(gray, 3)
lap = ed.laplacian(gray)

(zc, stdev) = ed.find_zero_cross(lap)
zc_filt = np.zeros(zc.shape, np.uint8)
zc *= 255
vals = np.sort(stdev, axis=None)
vals = [val for val in vals if val != 0]

ind = int(filt_ratio*len(vals) - 1)
th = vals[ind]
print(f'TH: {th}')
for i in range(0, zc.shape[0]):
    for j in range(0, zc.shape[1]):
        if(stdev[i, j] > th):
            zc_filt[i, j] = 255

ed.print_to_file(gray, 'output/gray.png')
ed.print_to_file(min_max_amp*ed.set_min_max(lap), 'output/minmax.png')
ed.print_to_file(zc_filt, 'output/edges_filt.png')
ed.print_to_file(zc, 'output/edges_all.png')