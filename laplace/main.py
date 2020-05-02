import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from edge_detect import EdgeDetect
from scipy import signal



infile = 'resources/face.png'
detect = EdgeDetect(infile)
# detect.averageFilter(15)
print('Filt done..')
detect.laplacian()
print('laplacian done..')
detect.find_zero_cross()

detect.print_to_file('output/face.png')