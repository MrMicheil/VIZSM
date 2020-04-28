import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


infile = 'resources/face1.jpg'
f, e = os.path.splitext(infile)
im = Image.open(infile)

arr = np.array(im)
shape = arr.shape

grey = np.zeros((shape[0], shape[1]), np.uint8)

print(f'Orig: {shape}, copy: {grey.shape}')

for i in range(0, shape[0]):
    for j in range(0, shape[1]):
        grey[i, j] = 0.3*arr[i, j, 0] + 0.59*arr[i,j,1] + 0.11*arr[i,j,2]

im2 = Image.fromarray(grey)
im2.save('output/grey1.jpg')