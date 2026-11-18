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



#----------------------Visibility------------------------

def visibilityWeightmap(img, ps):

    # Step 1: Gaussian blur
    blur = imgaussfilt(img, fsize=ps)

    # Step 2: Squared difference
    T = cv.pow(cv.subtract(img, blur), 2)

    # Step 3: Smooth the difference image
    T2 = imgaussfilt(T, sigma=0.5)

    # Step 4: Convert to HSV and extract Value channel
    hsv = cv.cvtColor(T2.astype(np.float32), cv.COLOR_RGB2HSV)
    V = hsv[:, :, 2]

    # Step 5: Local mean (replaces border_in + colfilt_mean)
    local_mean = cv.blur(
        V,
        (ps, ps),
        borderType=cv.BORDER_REPLICATE
    )

    # Step 6: Local absolute deviation
    diff = cv.absdiff(V, local_mean)

    # Step 7: Local sum (replaces colfilt_sum)
    vis = cv.boxFilter(
        diff,
        ddepth=-1,
        ksize=(ps, ps),
        normalize=False,
        borderType=cv.BORDER_REPLICATE
    )

    # Step 8: Normalize
    vis = cv.normalize(
        vis,
        None,
        0,
        1,
        cv.NORM_MINMAX
    )

    # Step 9: Prevent zero weights
    vis = vis + 0.01

    return vis




#-----------------------Chromaticity-----------------------
def chromaticWeightmap(img,ps):

    hsvImage = cv.cvtColor(img,cv.COLOR_RGB2HSV)
    hsvImage = cv.normalize(hsvImage.astype('double'),None,0.0,1.0,cv.NORM_MINMAX)
    r,c,_ = hsvImage.shape
    row = int(np.floor(r/ps))
    col = int(np.floor(c/ps))
    hsvImage = cv.resize(hsvImage,(col*ps,row*ps))
    r1,c1,_ = hsvImage.shape
    saturationValue = hsvImage[:,:,1]
    cW = np.zeros((r1,c1))
    #print(cW.dtype)

    sigma = 0.7

    for i in np.arange(0,r1-ps+1,ps):
        for j in np.arange(0,c1-ps+1,ps):
            s = saturationValue[i:i+ps,j:j+ps]
            smax = np.max(np.maximum(s,2),axis=0)[np.newaxis]
            temp = np.exp(-1 * (np.square(s-smax) / (2 * (sigma**2))))
            #print(temp.dtype)
            cW[i:i+ps,j:j+ps] = temp+0.1
    cW = cv.normalize(cW,None,0.0,1.0,cv.NORM_MINMAX)
    return cW



#--------------------------Colourfulnes------------------------

def colorfulnessWeightmap(img, ps):

    img = cv.normalize(img.astype(np.float32),
                       None, 0, 1,
                       cv.NORM_MINMAX)

    R, G, B = cv.split(img)

    rg = R - G
    by = (R + G) / 2 - B

    # Local mean
    mean_rg = cv.blur(rg, (ps, ps), borderType=cv.BORDER_REPLICATE)
    mean_by = cv.blur(by, (ps, ps), borderType=cv.BORDER_REPLICATE)

    # Local standard deviation (RG)
    mean_sq_rg = cv.blur(rg**2,(ps, ps),borderType=cv.BORDER_REPLICATE)

    variance_rg = mean_sq_rg - mean_rg**2
    variance_rg = np.maximum(variance_rg, 0)   # Prevent negative values

    std_rg = np.sqrt(variance_rg)

    # Local standard deviation (BY)

    mean_sq_by = cv.blur( by**2,(ps, ps),borderType=cv.BORDER_REPLICATE )
    variance_by = mean_sq_by - mean_by**2
    variance_by = np.maximum(variance_by, 0)
    std_by = np.sqrt(variance_by)

    # Colorfulness Measure
    CF = np.sqrt(std_rg**2 + std_by**2) + 0.3 * np.sqrt(mean_rg**2 + mean_by**2)
    CF += 0.01
    CF = cv.normalize(CF,None,0.0, 1.0, cv.NORM_MINMAX)

    return CF



#------------------LUMINANCE-------------------------
def luminanceWeightmap(img):

    img = cv.normalize(img,None,0.0,1.0,cv.NORM_MINMAX)
    R,G,B = cv.split(img)
    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(R)

    temp = np.sqrt((1/3) * np.square(R - R_mean) + np.square(G_mean - G) + np.square(B_mean - B))
    Max = np.max(np.max(temp,0))
    lW = temp/Max
    return lW


# White Balanced


# Contrast Enhanced

#--------------------------Visual Saliency-------------------
def imgaussfilt(img, sigma=0.5,fsize=None):
    """
    MATLAB imgaussfilt equivalent.
    """

    k = int(2 * np.ceil(2 * sigma) + 1)

    return cv.GaussianBlur(
        img,
        (k, k),
        sigmaX=sigma,
        borderType=cv.BORDER_REPLICATE
    )


def repmat(A, m, n):
    """
    MATLAB repmat equivalent.
    """

    return np.tile(A, (m, n))


def log_gabor(rows, cols, omega0=0.002, sigmaF=6.2):
    """
    Log-Gabor filter.

    Used for the frequency component of the
    visual saliency weight map.
    """

    # ---------------------------------------------------
    # Coordinate system centred at the image centre
    # ---------------------------------------------------

    x = np.linspace(
        -(cols // 2),
        cols // 2,
        cols
    )

    y = np.linspace(
        -(rows // 2),
        rows // 2,
        rows
    )

    X, Y = np.meshgrid(x, y)

    # ---------------------------------------------------
    # Radius from the centre
    # ---------------------------------------------------

    radius = np.sqrt(
        X**2 +
        Y**2
    )

    # Prevent log(0)
    radius[radius == 0] = 1

    # ---------------------------------------------------
    # Normalize radius
    # ---------------------------------------------------

    radius = radius / radius.max()

    # ---------------------------------------------------
    # Remove frequencies outside the unit circle
    # ---------------------------------------------------

    mask = radius <= 1

    radius = radius * mask

    # Prevent log(0) again
    radius[radius == 0] = 1

    # ---------------------------------------------------
    # Shift origin to FFT convention
    # ---------------------------------------------------

    radius = np.fft.ifftshift(radius)

    # ---------------------------------------------------
    # Log-Gabor Equation
    # ---------------------------------------------------

    LG = np.exp(
        -(
            np.log(radius / omega0)
        )**2
        /
        (
            2 *
            (
                np.log(sigmaF)
            )**2
        )
    )

    # Remove DC component
    LG[0, 0] = 0

    return LG


def visual_saliency_weight(
        img,
        sigmaF=6.2,
        omega0=0.002,
        sigmaD=114,
        sigmaC=0.25
):
    """
    Visual Saliency Weight Map

    Based on Equation (2) of the paper and
    """

    # -----------------------------------
    # Store original image size
    # -----------------------------------

    original_rows, original_cols = img.shape[:2]

    # -----------------------------------
    # Resize image to 256×256
    # -----------------------------------

    dsImage = cv.resize(
        img,
        (256, 256),
        interpolation=cv.INTER_LINEAR
    )

    # -----------------------------------
    # Convert to LAB
    # -----------------------------------

    lab = cv.cvtColor(
        dsImage,
        cv.COLOR_BGR2LAB
    ).astype(np.float32)

    # Normalize LAB image to [0,1]
    lab = cv.normalize(
        lab,
        None,
        0.0,
        1.0,
        cv.NORM_MINMAX
    )

    # -----------------------------------
    # Split LAB channels
    # -----------------------------------

    LChannel, AChannel, BChannel = cv.split(lab)

    # -----------------------------------
    # Fast Fourier Transform
    # -----------------------------------

    LFFT = np.fft.fft2(LChannel)
    AFFT = np.fft.fft2(AChannel)
    BFFT = np.fft.fft2(BChannel)

    rows, cols = dsImage.shape[:2]

    # -----------------------------------
    # Log-Gabor filter
    # -----------------------------------

    LG = log_gabor(
        rows,
        cols,
        omega0,
        sigmaF
    )

    # -----------------------------------
    # Apply Log-Gabor filter
    # -----------------------------------

    FinalLResult = np.real(
        np.fft.ifft2(LFFT * LG)
    )

    FinalAResult = np.real(
        np.fft.ifft2(AFFT * LG)
    )

    FinalBResult = np.real(
        np.fft.ifft2(BFFT * LG)
    )

    # -----------------------------------
    # Spectral Feature Map (SFMap)
    # -----------------------------------

    SFMap = np.sqrt(
        FinalLResult**2 +
        FinalAResult**2 +
        FinalBResult**2
    )

    # -----------------------------------
    # Colour Saliency Map (SCMap)
    # -----------------------------------

    labDistSquare = (
        AChannel**2 +
        BChannel**2
    )

    SCMap = (
        1.0 -
        np.exp(
            -labDistSquare /
            (sigmaC**2)
        )
    )

    # -----------------------------------
    # NOTE
    # -----------------------------------
    #
    # Next we will implement:
    #
    # 1. SDMap (Spatial Distribution Map)
    # 2. VSMap = SFMap × SCMap × SDMap
    # 3. Gaussian smoothing
    # 4. Resize to original image size
    # 5. Normalize
    # 6. +0.01

    # -----------------------------------
# Spatial Distribution Map (SDMap)
# -----------------------------------

    coordinateMtx = np.zeros(
        (rows, cols, 2),
        dtype=np.float32
    )

    a1 = np.arange(
        rows,
        dtype=np.float32
    )

    a2 = np.arange(
        cols,
        dtype=np.float32
    )

    coordinateMtx[:, :, 0] = np.tile(
        a1,
        (cols, 1)
    ).T

    coordinateMtx[:, :, 1] = np.tile(
        a2,
        (rows, 1)
    )

    centerY = rows / 2.0
    centerX = cols / 2.0

    centerMtx = np.ones(
        (rows, cols, 2),
        dtype=np.float32
    )

    centerMtx[:, :, 0] *= centerY
    centerMtx[:, :, 1] *= centerX

    SDMap = np.exp(
        -np.sum(
            (coordinateMtx - centerMtx) ** 2,
            axis=2
        )
        /
        (sigmaD ** 2)
    )

    VSMap = SFMap * SCMap * SDMap

    VSMap = cv.resize(
        VSMap,
        (original_cols, original_rows),
        interpolation=cv.INTER_LINEAR
    )

    VSMap = cv.GaussianBlur(
        VSMap,
        (11, 11),
        sigmaX=0.5,
        borderType=cv.BORDER_REPLICATE
    )

    VSMap = cv.normalize(
        VSMap,
        None,
        0.0,
        1.0,
        cv.NORM_MINMAX
    )

    VSMap += 0.01

    return VSMap.astype(np.float32)



#----------------------Entropy-------------
def entropy_weight(img, patch_size=11):

    """
    Local entropy weight map
    Based on Equation (4) of the paper.
    """

    gray = cv.cvtColor(
        img,
        cv.COLOR_BGR2GRAY
    )

    # Ensure the window size is odd
    if patch_size % 2 == 0:
        patch_size += 1

    ent = entropy(
        gray,
        square(patch_size)
    )

    ent = cv.normalize(
        ent.astype(np.float32),
        None,
        0.0,
        1.0,
        cv.NORM_MINMAX
    )

    # Avoid zero weights
    ent = ent + 0.01

    return ent.astype(np.float32)




#-----------------------Pyramid Fushion-----------------------------------------------


def pyr_reduce(img):

    kernel_width = 5
    cw = 0.5
    kerld = np.array([0.25-cw/2, 0.25, cw, 0.25, 0.25-cw/2])[np.newaxis]
    kernel = np.kron(kerld,kerld.T)

    img = cv.normalize(img.astype('double'),None,0.0,1.0,cv.NORM_MINMAX)
    sz = img.shape
    #print(sz)
    #imgout = np.zeros(((np.floor(sz[0]/2)).astype('uint8'),(np.floor(sz[1]/2)).astype('uint8'),sz[2]))
    sz2 = [(np.ceil(sz[0]/2).astype('uint16')),(np.ceil(sz[1]/2).astype('uint16'))]
    #print(sz2)
    if img.ndim == 3:
        imgout = np.zeros((sz2[0],sz2[1],sz[2]))
        for p in range(sz[2]):
            img1 = img[:,:,p]
            imgFiltered = correlate(img1,kernel,mode='nearest')
            #x = np.arange(0,sz[0],2)
            #y = np.arange(0,sz[1],2)
            #print(x.shape,y.shape)
            imgout[:,:,p] = imgFiltered[0:sz[0]:2,0:sz[1]:2]

        return np.array(imgout)
    else:
        imgFiltered = correlate(img,kernel,mode='nearest')
        imgout = imgFiltered[0:sz[0]:2,0:sz[1]:2]

        return np.array(imgout)

def pyr_expand(img):

    kw = 5
    cw = 0.5
    kerld = np.array([0.25-cw/2, 0.25, cw, 0.25, 0.25-cw/2])[np.newaxis]
    kernel = np.kron(kerld,kerld.T) * 4

    ker00 = kernel[0:kw:2,0:kw:2]
    ker01 = kernel[0:kw:2,1:kw:2]
    ker10 = kernel[1:kw:2,0:kw:2]
    ker11 = kernel[1:kw:2,1:kw:2]

    img = cv.normalize(img.astype('double'),None,0.0,1.0,cv.NORM_MINMAX)
    sz = img.shape
    osz = [sz[0]*2-1,sz[1]*2-1]

    if img.ndim == 3:
        imgout = np.zeros((osz[0],osz[1],sz[2]))

        for p in range(sz[2]):
            img1 = img[:,:,p]
            img1ph = np.pad(img1,(0,1),'edge')
            img1pv = np.pad(img1,(1,0),'edge')

            img00 = correlate(img1,ker00,mode='nearest')
            img01 = convolve2d(img1ph,ker01,'valid')
            img10 = convolve2d(img1pv,ker10,'valid')
            img11 = convolve2d(img1,ker11,'valid')

            imgout[0:osz[0]:2,0:osz[1]:2,p] = img00
            imgout[0:osz[0]:2,1:osz[1]:2,p] = img10
            imgout[1:osz[0]:2,0:osz[1]:2,p] = img01
            imgout[1:osz[0]:2,1:osz[1]:2,p] = img11

        return imgout
    else:
        imgout = np.zeros((osz[0],osz[1]))

        img1 = img
        img1ph = np.pad(img1,(0,1),'edge')
        img1pv = np.pad(img1,(1,0),'edge')

        img00 = correlate(img1,ker00,mode='nearest')
        img01 = convolve2d(img1ph,ker01,'valid')
        img10 = convolve2d(img1pv,ker10,'valid')
        img11 = convolve2d(img1,ker11,'valid')

        imgout[0:osz[0]:2,0:osz[1]:2] = img00
        imgout[0:osz[0]:2,1:osz[1]:2] = img10
        imgout[1:osz[0]:2,0:osz[1]:2] = img01
        imgout[1:osz[0]:2,1:osz[1]:2] = img11

        return imgout
#--------------------------------------GAUSSIAN PYRAMID--------------------
def genPyr(img,typ,level):
    
    pyr = []
    
    pyr.append(cv.normalize(img.astype('double'),None,0.0,1.0,cv.NORM_MINMAX))
    
    for i in range(1,level):
        pyr.append(pyr_reduce(pyr[i-1]))
        
    if typ == 'gauss':
        return pyr
    
    for p in reversed(range(0,level-1)):
        osz = [pyr[p+1].shape[0]*2-1,pyr[p+1].shape[1]*2-1]
        if pyr[p].ndim == 3:
            pyr[p] = pyr[p][0:osz[0],0:osz[1],:]
        else:
            pyr[p] = pyr[p][0:osz[0],0:osz[1]]
            
    for p in range(0,level-1):
        pyr[p] = pyr[p]-pyr_expand(pyr[p+1])
        
    return pyr

#----------------------------------------RECONSTRUCT PYRAMID------------------------

def pyrReconstruct(pyr):

    for p in np.arange(len(pyr)-2,-1,-1):
        exp = pyr_expand(pyr[p+1])
        exp = cv.resize(exp,(pyr[p].shape[1],pyr[p].shape[0]))
        #print(pyr[p].shape,exp.shape)
        pyr[p] = pyr[p]+exp

    return pyr[0]
#=========================================DEHAZING=========================================
def pyramidFusionDehaze(img,ps):

    firstInput = whiteBalance(img)
    secondInput = enhanceContrast(img)
    ps = ps
    row,col,_ = img.shape

    visibilityWeightmap1 = visibilityWeightmap(firstInput,ps)
    saliencyWeightmap1 = visual_saliency_weight(firstInput)
    entropyWeightmap1 = entropy_weight(firstInput,ps)
    chromaticWeightmap1 = chromaticWeightmap(firstInput,ps)
    colorfulnessWeightmap1 = colorfulnessWeightmap(firstInput,ps)
    luminanceWeightmap1 = luminanceWeightmap(firstInput)

    r1,c1 = entropyWeightmap1.shape

    visibilityWeightmap1 = cv.resize(visibilityWeightmap1,(c1,r1))
    saliencyWeightmap1 = cv.resize(saliencyWeightmap1,(c1,r1))
    entropyWeightmap1 = cv.resize(entropyWeightmap1,(c1,r1))
    chromaticWeightmap1 = cv.resize(chromaticWeightmap1,(c1,r1))
    colorfulnessWeightmap1 = cv.resize(colorfulnessWeightmap1,(c1,r1))
    luminanceWeightmap1 = cv.resize(luminanceWeightmap1,(c1,r1))

    visibilityWeightmap2 = visibilityWeightmap(secondInput,ps)
    saliencyWeightmap2 = visual_saliency_weight(secondInput)
    entropyWeightmap2 = entropy_weight(secondInput,ps)
    chromaticWeightmap2 = chromaticWeightmap(secondInput,ps)
    colorfulnessWeightmap2 = colorfulnessWeightmap(secondInput,ps)
    luminanceWeightmap2 = luminanceWeightmap(secondInput)

    r2,c2 = entropyWeightmap2.shape

    visibilityWeightmap2 = cv.resize(visibilityWeightmap2,(c1,r1))
    saliencyWeightmap2 = cv.resize(saliencyWeightmap2,(c1,r1))
    entropyWeightmap2 = cv.resize(entropyWeightmap2,(c1,r1))
    chromaticWeightmap2 = cv.resize(chromaticWeightmap2,(c1,r1))
    colorfulnessWeightmap2 = cv.resize(colorfulnessWeightmap2,(c1,r1))
    luminanceWeightmap2 = cv.resize(luminanceWeightmap2,(c1,r1))

    Visibility1 = np.divide(visibilityWeightmap1,(visibilityWeightmap1+visibilityWeightmap2))
    Visibility2 = np.divide(visibilityWeightmap2,(visibilityWeightmap1+visibilityWeightmap2))
    Saliency1 = np.divide(saliencyWeightmap1,(saliencyWeightmap1+saliencyWeightmap2))
    Saliency2 = np.divide(saliencyWeightmap2,(saliencyWeightmap1+saliencyWeightmap2))
    Entropy1 = np.divide(entropyWeightmap1,(entropyWeightmap1+entropyWeightmap2))
    Entropy2 = np.divide(entropyWeightmap2,(entropyWeightmap1+entropyWeightmap2))
    Chromatic1 = np.divide(chromaticWeightmap1,(chromaticWeightmap1+chromaticWeightmap2))
    Chromatic2 = np.divide(chromaticWeightmap2,(chromaticWeightmap1+chromaticWeightmap2))
    Color1 = np.divide(colorfulnessWeightmap1,(colorfulnessWeightmap1+colorfulnessWeightmap2))
    Color2 = np.divide(colorfulnessWeightmap2,(colorfulnessWeightmap1+colorfulnessWeightmap2))
    Luminance1 = np.divide(luminanceWeightmap1,(luminanceWeightmap1+luminanceWeightmap2))
    Luminance2 = np.divide(luminanceWeightmap2,(luminanceWeightmap1+luminanceWeightmap2))

    resultedWeightmap1 =   Saliency1 * Visibility1 * Entropy1 * Chromatic1 * Luminance1 * Color1
    resultedWeightmap2 =   Saliency2 * Visibility2 * Entropy2 * Chromatic2 * Luminance2 * Color2

    normalizedWeightmap1 = np.divide(resultedWeightmap1,(resultedWeightmap1+resultedWeightmap2))
    normalizedWeightmap2 = np.divide(resultedWeightmap2,(resultedWeightmap1+resultedWeightmap2))

    gaussianPyramid1 = genPyr(normalizedWeightmap1,'gauss',5)
    gaussianPyramid2 = genPyr(normalizedWeightmap2,'gauss',5)

    fusedPyramid = []
    for i in range(0,5):
        tempImg = []
        for j in range(0,3):
            laplacianPyramid1 = genPyr(firstInput[:,:,j],'laplacian',5)
            laplacianPyramid2 = genPyr(secondInput[:,:,j],'laplacian',5)
            rowSize = np.min([laplacianPyramid1[i].shape[0],laplacianPyramid2[i].shape[0],gaussianPyramid1[i].shape[0],gaussianPyramid2[i].shape[0]])
            columnSize = np.min([laplacianPyramid1[i].shape[1],laplacianPyramid2[i].shape[1],gaussianPyramid1[i].shape[1],gaussianPyramid2[i].shape[1]])
            fti = np.multiply(laplacianPyramid1[i][0:rowSize,0:columnSize],gaussianPyramid1[i][0:rowSize,0:columnSize])
            sti = np.multiply(laplacianPyramid2[i][0:rowSize,0:columnSize],gaussianPyramid2[i][0:rowSize,0:columnSize])
            tempImg.append((fti+sti))
        fusedPyramid.append(np.dstack((tempImg[0],tempImg[1],tempImg[2])))

    enhancedImg = pyrReconstruct(fusedPyramid)
    enhancedImg = imadjust(enhancedImg,enhancedImg.min(),enhancedImg.max())

    return enhancedImg



# ================= Batch Processing ==============
import os

input_root = "/home/priyasha/multi_gt_images"
output_root = "/home/priyasha/multi_gt_images_dehazed"

os.makedirs(output_root, exist_ok=True)

for folder in os.listdir(input_root):

    input_folder = os.path.join(input_root, folder)

    # Skip anything that's not a folder
    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)

    for f in os.listdir(input_folder):

        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_path = os.path.join(input_folder, f)

        img = cv.imread(input_path)

        if img is None:
            print("Couldn't read:", input_path)
            continue

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        dehazed = pyramidFusionDehaze(img, 11)

        plt.imsave(
            os.path.join(
                output_folder,
                os.path.splitext(f)[0] + "_d.png"
            ),
            dehazed
        )

        print(f"Processed: {folder}/{f}")

print("All images processed successfully!")
