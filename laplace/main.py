import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from edge_detect import EdgeDetect


infile = 'resources/sudoku.png'
detect = EdgeDetect(infile)
detect.averageFilter(15)
detect.laplacian()
detect.find_zero_cross()

detect.print_to_file('output/face.png')