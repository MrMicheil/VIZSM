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

rgb = ed.load_image(infile) # nacitat RGB
gray = ed.to_gray(rgb)      # do greyscale
gray = ed.averageFilter(gray, 5)    # filtre
gray = ed.median_filter(gray, 3)
lap = ed.laplacian(gray)        # laplacian

(zc, stdev) = ed.find_zero_cross(lap)   # prechod nulou, lok. rozptyl
zc *= 255

# urcenie hranicnej hodnoty
zc_filt = np.zeros(zc.shape, np.uint8)
vals = np.sort(stdev, axis=None)    # triedenie
vals = [val for val in vals if val != 0]    # vymazat 0-y

ind = int(filt_ratio*len(vals) - 1) # vyber hranicnej hodnoty
th = vals[ind]
print(f'TH: {th}')

# filtrovanie hran
for i in range(0, zc.shape[0]):
    for j in range(0, zc.shape[1]):
        if(stdev[i, j] > th):
            zc_filt[i, j] = 255

mm = min_max_amp*ed.set_min_max(lap) # obmedzenie hodnot + zosilnenie

ed.print_to_file(gray, 'output/gray.png')
ed.print_to_file(mm, 'output/minmax.png')
ed.print_to_file(zc_filt, 'output/edges_filt.png')
ed.print_to_file(zc, 'output/edges_all.png')
