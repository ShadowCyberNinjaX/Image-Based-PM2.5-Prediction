import cv2 as cv
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import pywt
from skimage.util import view_as_windows as viewW
from skimage.morphology import rectangle,square
from skimage.filters.rank import entropy
from scipy.ndimage import correlate,gaussian_filter
from scipy.signal import convolve2d
import os
import pandas as pd
from skimage.metrics import structural_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import joblib as jb
from sklearn.decomposition import PCA, KernelPCA
from PIL import Image
import datetime

im = cv.imread('/home/priyasha/data process/2026-04-12/001/jpg/10/03/2026-04-12_10_03_View1.jpg')
im = cv.cvtColor(im,cv.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(im)

def whiteBalance(img):
    img = cv.normalize(img.astype('double'),None,0.0,1.0,cv.NORM_MINMAX)
    r,g,b = cv.split(img)
    h,w,c = img.shape

    whiteBalanced = np.zeros((h,w,c))

    r_avg = np.average(r)
    g_avg = np.average(g)
    b_avg = np.average(b)

    gray_value = (r_avg + g_avg + b_avg)/3
    scale_value_1 = gray_value/r_avg
    scale_value_2 = gray_value/g_avg
    scale_value_3 = gray_value/b_avg

    whiteBalanced[:,:,0] = scale_value_1 * r
    whiteBalanced[:,:,1] = scale_value_2 * g
    whiteBalanced[:,:,2] = scale_value_3 * b

    whiteBalanced = np.multiply(np.divide(whiteBalanced - np.min(whiteBalanced),(np.max(whiteBalanced) - np.min(whiteBalanced))),255)
    whiteBalanced = whiteBalanced.astype(np.uint8)

    return whiteBalanced

wb = whiteBalance(im)

plt.figure(figsize=(10,10))
plt.imshow(wb)


#------------------CONTRAST ENHANCEMENT--------------------
def imadjust(x,a,b,c=0,d=1,gamma=1):

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c

    return y

def imgaussfilt(img,sigma=0.5,fsize='def',pad=cv.BORDER_REPLICATE):

    if fsize=='def':
        k = 2*np.ceil(2*sigma)+1
    else:
        k = fsize
    #print(k)
    k = int(k)
    j = cv.GaussianBlur(img,ksize=(k,k),sigmaX=sigma,borderType=pad)
    return j


#---WAVELET TRANSFORM-------

def fuseCoeff(c1,c2,meth):

    if meth == 'mean':
        cf = (c1+c2)/2
    elif meth == 'max':
        cf = np.maximum(c1,c2)
    elif meth == 'min':
        cf = np.minimum(c1,c2)
    else:
        cf = []

    return cf

def wFuse(im1,im2,wav,mode,lvl,ameth,dmeth):

    coeff1 = pywt.wavedec2(im1[:,:],wav,mode,lvl)
    coeff2 = pywt.wavedec2(im2[:,:],wav,mode,lvl)

    fusedCoeff = []
    for i in range(len(coeff1)):

        if i==0:
            fusedCoeff.append(fuseCoeff(coeff1[i],coeff2[i],ameth))
        else:
            c1 = fuseCoeff(coeff1[i][0],coeff2[i][0],dmeth)
            c2 = fuseCoeff(coeff1[i][1],coeff2[i][1],dmeth)
            c3 = fuseCoeff(coeff1[i][2],coeff2[i][2],dmeth)

            fusedCoeff.append((c1,c2,c3))

        fusedImage = pywt.waverec2(fusedCoeff,wav,mode)
        fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
        fusedImage = fusedImage.astype(np.uint8)

        return fusedImage

def wfusimg(im1,im2,wav,mode,lvl,ameth,dmeth):
    
    fusedImage = []
    if im1.ndim == 3 & im2.ndim == 3:
        for i in range(3):
            im = wFuse(im1[:,:,i],im2[:,:,i],wav,mode,lvl,ameth,dmeth)
            fusedImage.append(im)
        xfus = np.dstack((fusedImage[0],fusedImage[1],fusedImage[2]))
    else:
        xfus = wFuse(im1,im2,wav,mode,lvl,ameth,dmeth)
        
    return xfus

#-----------------------------------------------------

def enhanceContrast(img):

    # Step 1: Convert to grayscale and normalize
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray = cv.normalize(gray.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # Step 2: Normalize RGB image
    img = cv.normalize(img.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # Step 3: Split RGB channels
    r, g, b = cv.split(img)

    # Step 4: Compute mean intensities
    mean_r = np.average(r)
    mean_g = np.average(g)
    mean_b = np.average(b)
    mean_gray = np.average(gray)

    # Step 5: Adjust each RGB channel
    r = r * (mean_r / mean_gray)
    g = g * (mean_g / mean_gray)
    b = b * (mean_b / mean_gray)

    # Step 6: Normalize each channel
    r = cv.normalize(r, None, 0.0, 1.0, cv.NORM_MINMAX)
    g = cv.normalize(g, None, 0.0, 1.0, cv.NORM_MINMAX)
    b = cv.normalize(b, None, 0.0, 1.0, cv.NORM_MINMAX)

    # Step 7: Merge channels
    adjusted_img = np.dstack((r, g, b))

    # ---------- Image I ----------
    # Contrast enhancement
    I = imadjust(adjusted_img,
                 adjusted_img.min(),
                 adjusted_img.max())

    # ---------- Image J ----------
    # Gaussian smoothing
    G = imgaussfilt(adjusted_img)

    # High-boost sharpening
    J = cv.add(adjusted_img,
               cv.subtract(adjusted_img, G))

    # Normalize sharpened image
    J = imadjust(J, J.min(), J.max())

    # ---------- Wavelet Fusion ----------
    fused = wfusimg(I,
                    J,
                    'sym4',
                    'symmetric',
                    0,
                    'max',
                    'max')

    # ---------- CLAHE ----------
    clahe = cv.createCLAHE(clipLimit=5,
                           tileGridSize=(8,8))

    r = clahe.apply(fused[:,:,0])
    g = clahe.apply(fused[:,:,1])
    b = clahe.apply(fused[:,:,2])

    # Step 8: Merge final channels
    enhanced = np.dstack((r, g, b))

    return enhanced

enh = enhanceContrast(im)

plt.figure(figsize=(12,12))
plt.imshow(enh)

plt.figure(figsize=(15,5))

# Original
plt.subplot(131)
plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

# White Balanced
plt.subplot(132)
plt.imshow(cv.cvtColor(wb, cv.COLOR_BGR2RGB))
plt.title("White Balanced")
plt.axis("off")

# Contrast Enhanced
plt.subplot(133)
plt.imshow(cv.cvtColor(enh, cv.COLOR_BGR2RGB))
plt.title("Contrast Enhanced")
plt.axis("off")

plt.tight_layout()
plt.savefig("contrast_enhancement.jpg", dpi=300, bbox_inches="tight")
plt.show()
