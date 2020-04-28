from PIL import Image
import numpy as np

class EdgeDetect:

    def __init__(self, path):

        self.rgb = np.array(Image.open(path))
        shape = self.rgb.shape

        self.grey = np.zeros((shape[0], shape[1]), np.uint8)
        self.w = shape[0]
        self.h = shape[1]
        
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                self.grey[i, j] =\
                    0.3*self.rgb[i, j, 0] + 0.59*self.rgb[i,j,1] + 0.11*self.rgb[i,j,2]
    
    def __vals_around(self, x, y, sz)->list:
        xmin = int(x - sz/2)
        ymin = int(y - sz/2)
        xmax = int(x + sz/2)
        ymax = int(y + sz/2)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.w, xmax)
        ymax = min(self.h, ymax)

        vals = []
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                vals.append(self.grey[i, j])
        
        return vals

    
    def __find_median(self, x, y, sz)->np.uint8:
        vals = self.__vals_around(x, y, sz)
        return np.median(vals)



    def median_filter(self, size):
        if(size % 2 == 0):
            raise ValueError("size shouldn't be even")
        cpy = np.copy(self.grey)

        for i in range(0, self.w):
            for j in range(0, self.h):
                cpy[i, j] = self.__find_median(i, j, size)

        self.grey = cpy


    def __find_average(self, x, y, sz)->float:
        vals = self.__vals_around(x, y, sz)
        return np.average(vals)


    def averageFilter(self, size):
        if(size % 2 == 0):
            raise ValueError("size shouldn't be even")

        cpy = np.copy(self.grey)
        for i in range(0, self.w):
            for j in range(0, self.h):
                cpy[i, j] = self.__find_average(i, j, size)

        self.grey = cpy


    def print_to_file(self, filename):
        Image.fromarray(self.grey).save(filename)
