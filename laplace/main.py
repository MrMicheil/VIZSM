import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import edge_detect as ed
from scipy import signal


infile = 'resources/simp.png'

im = Image.open(infile)
arr = np.array(im)
gray = ed.to_gray(arr)

lap = ed.laplacian(gray)
norm = ed.set_min_max(lap)
zc = ed.find_zero_cross(lap)
zc *= 255
ed.print_to_file(zc, 'output/gray.png')
ed.print_to_file(norm, 'output/norm.png')