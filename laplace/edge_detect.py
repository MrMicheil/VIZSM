from PIL import Image
import numpy as np
from scipy import signal


def sign(num):
    if(num >= 0):
        return 1
    elif(num < 0):
        return -1    

class EdgeDetect:

    def __init__(self, path):

        self.rgb = np.array(Image.open(path))
        shape = self.rgb.shape

        self.grey = np.zeros((shape[0], shape[1]), np.uint8)
        self.laplace = np.zeros((shape[0], shape[1]), np.int)
        self.w = shape[0]
        self.h = shape[1]
        
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                self.grey[i, j] =\
                     0.299*self.rgb[i, j, 0] + 0.587*self.rgb[i,j,1] + 0.114*self.rgb[i,j,2]

    def __get_safe_inds(self, x, y, sz):
        xmin = int(x - sz/2)
        ymin = int(y - sz/2)
        xmax = int(x + sz/2)
        ymax = int(y + sz/2)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.w-1, xmax+1)
        ymax = min(self.h-1, ymax+1)

        return ((xmin, xmax), (ymin, ymax))
    
    def __vals_around(self, x, y, sz)->list:
        
        x_safe, y_safe = self.__get_safe_inds(x, y, sz)

        vals = []
        for i in range(x_safe[0], x_safe[1]+1):
            for j in range(y_safe[0], y_safe[1]+1):
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


    def averageFilter(self, size):
        if(size % 2 == 0):
            raise ValueError("size shouldn't be even")
        mat = np.full((size, size), 1/(size*size), np.float)
        self.grey = self.__correlate(mat).astype(np.uint8)


    def __valid_ind_x(self, ind)->bool:
        return ind > 0 and ind < self.w
    
    def __valid_ind_y(self, ind)->bool:
        return ind > 0 and ind < self.h


    def __correlate(self, mat):
        sz = mat.shape
        if(sz[0] != sz[1]):
            raise ValueError("mat should be square")
        if(sz[0] % 2 == 0):
            raise ValueError("mat size shouldn't be even")
        return signal.correlate2d(self.grey.astype(int), mat, boundary='fill', mode='full')


    
    def laplacian(self):
        mat1 = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])
        mat2 = np.array([[1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]])
        self.laplace = self.__correlate(mat1)
        return
        for i in range(0, self.grey.shape[0]):
            for j in range(0, self.grey.shape[1]):
                if self.laplace[i,j] < 0:
                    self.grey[i, j] = 0
                elif self.laplace[i, j] > 255/5:
                    self.grey[i, j] = 255
                else:
                    self.grey[i, j] = 5*self.laplace[i, j]

    
    
    def __calc_mf(self, x, y, sz):
        x_safe, y_safe = self.__get_safe_inds(x, y, sz)

        coeff = 1/((2*sz + 1)**2)
        sum = 0
        for k1 in range(x_safe[0], x_safe[1]+1):
            for k2 in range(y_safe[0], y_safe[1] + 1):
                sum += self.grey[k1, k2]
        return coeff*sum

    def __calc_deviation(self, x, y, sz):
        x_safe, y_safe = self.__get_safe_inds(x, y, sz)

        coeff = 1/((2*sz + 1)**2)
        sum = 0
        for k1 in range(x_safe[0], x_safe[1]+1):
            for k2 in range(y_safe[0], y_safe[1] + 1):
                sum += (self.grey[k1, k2] - self.__calc_mf(k1, k2, sz))**2
        return coeff * sum
    
    def __check_ZC(self, x, y):
        lap = self.laplace

        if(lap[x-1, y] * lap[x, y] < 0):
            return True
        elif(lap[x, y-1] * lap[x, y] < 0):
            return True
        else:
            return False

        if(lap[x-1, y-1]*lap[x+1, y+1] < 0):
            return True
        if(lap[x-1, y]*lap[x+1, y] < 0):
            return True
        if(lap[x-1, y+1]*lap[x+1, y-1] < 0):
            return True
        if(lap[x, y-1]*lap[x, y+1] < 0):
            return True
        return False


    def find_zero_cross(self):
        self.edges = np.zeros((self.w, self.h), np.uint8)
        for i in range(1, self.w - 2, 1):
            for j in range(1, self.h - 2, 1):
                if(self.__check_ZC(i, j)):
                    #dev = self.__calc_deviation(i, j, 5)
                    #if(dev > 5000):
                    self.edges[i,j] = 255
            print(i)
        
        self.grey = self.edges
                
                


    def print_to_file(self, filename):
        Image.fromarray(self.grey).save(filename)
