import numpy as np
import tensorflow as tf


##Function: original image + alpha in (0,1)!
###Return: Range lower bound on max mean
def max_mean(original, alpha):
    beta = (1-alpha)/alpha
    u_o = np.mean(original)
    #range_x = np.max(original)
    c1 = (0.01)**2
    u_max = beta*u_o+np.sqrt(beta**2*u_o**2+beta*(c1+u_o**2))
    u_min = beta*u_o-np.sqrt(beta**2*u_o**2+beta*(c1+u_o**2))
    return u_min, u_max

###Function: original (gray) image + eta in (0,1)
###Out: approx on max allowed variance
def max_var(original, eta):
    gamma = (1-eta)/eta
    W = np.size(original, 0)
    H = np.size(original, 1)
    c2 = (0.03)**2
    mean = np.mean(original)
    var_or = np.sum((original-mean)**2)/(W*H-1)
    M = c2 + 2*var_or
    return M*gamma

def low_std_dev(original, eta):
    gamma = (1-eta)/eta
    c2 = (0.03)**2
    W = np.size(original, 0)
    H = np.size(original, 1)
    mean = np.mean(original)
    var_or = np.sum((original-mean)**2)/(W*H-1)
    std_or = np.sqrt(var_or)
    res = gamma*std_or+np.sqrt(gamma**2*var_or+gamma*(2*var_or+c2))
    return res

###function: original gray image, perturbed gray image
###Out: True SSIM score
def SSIM(original, perturbed):
    mean_or = np.mean(original)
    mean_per = np.mean(perturbed)
    W = np.size(original, 0)
    H = np.size(original, 1)
    c1 = (0.01)**2
    c2 = (0.01*3)**2
    var_o = np.sum(((original-mean_or)**2)/(W*H-1))
    var_per = np.sum(((perturbed-mean_per)**2)/(W*H-1))
    cov = np.sum((original-mean_or)*(perturbed-mean_per)/(W*H-1))
    SSIM = (2*mean_or*mean_per+c1)*(2*cov+c2)/((mean_or**2+mean_per**2+c1)*(var_o+var_per+c2))
    return SSIM

###Convert rgb image to grayscale, taking luminance perception into account
###In: rgb image
def rgb2gray(rgb):
    r = rgb[:,:, 0]
    g = rgb[:,:, 1]
    b = rgb[:,:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


###Intuitive implimentation gray to rgb
def to_rgb(delta, im):
    # I think this will be slow
    w,h,RGB = im.shape
    ret = np.empty((w, h, RGB), dtype=np.float32)
    ret[:, :, 0] = im[:,:,0]+delta
    ret[:, :, 1] = im[:,:,1]+delta
    ret[:, :, 2] = im[:,:,2]+delta
    return ret
