from PIL import Image
import numpy as np


def sign(num):
    if(num > 0):
        return 1
    elif(num < 0):
        return -1
    else:
        return 0    

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
                    0.3*self.rgb[i, j, 0] + 0.59*self.rgb[i,j,1] + 0.11*self.rgb[i,j,2]
    
    def __vals_around(self, x, y, sz)->list:
        xmin = int(x - sz/2)
        ymin = int(y - sz/2)
        xmax = int(x + sz/2)
        ymax = int(y + sz/2)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.w-1, xmax+1)
        ymax = min(self.h-1, ymax+1)

        vals = []
        for i in range(xmin, xmax+1):
            for j in range(ymin, ymax+1):
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
        mat = np.full((size, size), 1/(size*size), np.float)
        self.grey = self.__convolute(mat)


    def __valid_ind_x(self, ind)->bool:
        return ind > 0 and ind < self.w
    
    def __valid_ind_y(self, ind)->bool:
        return ind > 0 and ind < self.h

    def __convoluteXY(self, x, y, mat)->float:
        sz = mat.shape[0]
        center = int(sz / 2)
        val = 0
        for i in range(-center, center+1):
            for j in range(-center, center+1):
                param = mat[i + center][j+center]
                x_ind = x + i
                y_ind = y + j
                curr = 0
                if(self.__valid_ind_x(x_ind) and self.__valid_ind_y(y_ind)):
                    curr = self.grey[x_ind, y_ind]
                val += param * curr
        return val


    def __convolute(self, mat):
        sz = mat.shape
        if(sz[0] != sz[1]):
            raise ValueError("mat should be square")
        if(sz[0] % 2 == 0):
            raise ValueError("mat size shouldn't be even")
        cpy = np.copy(self.grey)
        self.__convoluteTo(mat, cpy)
        return cpy
    

    def __convoluteTo(self, mat, dest):
        sz = mat.shape
        if(sz[0] != sz[1]):
            raise ValueError("mat should be square")
        if(sz[0] % 2 == 0):
            raise ValueError("mat size shouldn't be even")

        for i in range(0, self.w):
            for j in range(0, self.h):
                dest[i, j] = self.__convoluteXY(i, j, mat)


    
    def laplacian(self):
        mat1 = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])
        mat1 = np.array([[1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]])
        self.__convoluteTo(mat1, self.laplace)

    
    
    def __check_ZC(self, x, y):
        lap = self.laplace
        start = sign(lap[x-1, y-1])
        for i in range(-1, 2):
            for j in range(-1, 2):
                if(sign(lap[x+i, y+1]) != start):
                    return True
        return False


    def find_zero_cross(self):
        self.edges = np.zeros((self.w, self.h), np.uint8)
        for i in range(1, self.w - 2):
            for j in range(1, self.h - 2):
                if(self.__check_ZC(i, j)):
                    self.edges[i,j] = 255
        
        self.grey = self.edges
                
                


    def print_to_file(self, filename):
        Image.fromarray(self.grey).save(filename)
