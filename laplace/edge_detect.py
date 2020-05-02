from PIL import Image
import numpy as np
from scipy import signal


def __sign(num):
    if(num > 0):
        return 1
    elif(num < 0):
        return -1
    else:
        return 0


  

def to_gray(mat):
    ret = np.empty(mat.shape[0:2], np.uint8)
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            ret[i, j] = 0.299*mat[i, j, 0] + 0.587*mat[i,j,1] + 0.114*mat[i,j,2]
    return ret


def correlate(mat1, mat2):
    return signal.correlate2d(mat1.astype(np.float), mat2)


def __get_safe_inds(mat, x, y, sz):
    xmin = int(x - sz/2)
    ymin = int(y - sz/2)
    xmax = int(x + sz/2)
    ymax = int(y + sz/2)

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(mat.shape[0]-1, xmax+1)
    ymax = min(mat.shape[1]-1, ymax+1)

    return ((xmin, xmax), (ymin, ymax))


def __vals_around(mat, x, y, sz)->list:
    
    x_safe, y_safe = __get_safe_inds(mat, x, y, sz)

    vals = []
    for i in range(x_safe[0], x_safe[1]+1):
        for j in range(y_safe[0], y_safe[1]+1):
            vals.append(mat[i, j])
    
    return vals


def __find_median(mat, x, y, sz):
    vals = __vals_around(mat, x, y, sz)
    return np.median(vals)



def median_filter(mat, size):
    if(size % 2 == 0):
        raise ValueError("size shouldn't be even")
    cpy = np.empty(mat.shape, np.uint8)
    w, h = mat.shape

    for i in range(0, w):
        for j in range(0, h):
            cpy[i, j] = __find_median(mat, i, j, size)

    return cpy


def averageFilter(mat, size):
    if(size % 2 == 0):
        raise ValueError("size shouldn't be even")
    mat2 = np.full((size, size), 1/(size*size), np.float)
    return correlate(mat, mat2).astype(np.uint8)


def laplacian(mat):
    mat1 = np.array([[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]])
    mat2 = np.array([[1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]])
    laplace = correlate(mat, mat1)

    return laplace
    
    

def set_min_max(mat, min=0, max=255):
    mat2 = np.empty(mat.shape, np.uint8)
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if mat[i,j] < min:
                mat2[i, j] = min
            elif mat[i, j] > max:
                mat2[i, j] = max
            else:
                mat2[i, j] = mat[i, j]
    return mat2

def __calc_mf(mat, x, y, sz):
    x_safe, y_safe = __get_safe_inds(mat, x, y, sz)

    coeff = 1/((2*sz + 1)**2)
    slc = mat[x_safe[0]:x_safe[1]+1, y_safe[0]:y_safe[1]+1]
    sum = np.sum(slc)
    return coeff*sum


def __calc_deviation(mat, x, y, sz):
    x_safe, y_safe = __get_safe_inds(mat, x, y, sz)

    coeff = 1/((2*sz + 1)**2)
    sum = 0
    for k1 in range(x_safe[0], x_safe[1]+1):
        for k2 in range(y_safe[0], y_safe[1] + 1):
            sum += (mat[k1, k2] - __calc_mf(mat, k1, k2, sz))**2
    return coeff * sum


def __check_ZC(mat, x, y):

    #diag1
    if(mat[x-1, y-1]*mat[x+1, y+1] < 0):
        return True
    #horiz
    if(mat[x-1, y]*mat[x+1, y] < 0):
        return True
    #diag2
    if(mat[x-1, y+1]*mat[x+1, y-1] < 0):
        return True
    return False


def find_zero_cross(mat):
    edges = np.zeros(mat.shape, np.uint8)
    stdev = np.zeros(mat.shape, np.float)
    w, h = mat.shape
    for i in range(1, w - 1, 1):
        for j in range(1, h - 1, 1):
            if(__check_ZC(mat, i, j)):
                stdev[i,j] = __calc_deviation(mat, i, j, 5)
                edges[i,j] = 1
        print(i)
    
    return (edges, stdev)
            
            


def print_to_file(mat, filename):
    Image.fromarray(mat.astype(np.uint8)).save(filename)
