import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from edge_detect import EdgeDetect


infile = 'resources/face1.jpg'

detect = EdgeDetect(infile)
detect.averageFilter(5)

detect.print_to_file('output/face1.jpg')