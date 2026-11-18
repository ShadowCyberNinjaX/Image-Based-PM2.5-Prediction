import os
import shutil
from pathlib import Path
from datetime import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from skimage.metrics import structural_similarity

pm = pd.read_csv('/home/priyasha/new data/frb(train).csv')

roi_1=[]
eroi_1=[]
roi_2=[]
eroi_2=[]
#oroi_1=[]
o=cv.imread('/home/priyasha/new data/roi_preset_combined_train_dehazed/01-Apr-2022_15_18_d.png')
#o_gray=cv.cvtColor(o,cv.COLOR_BGR2GRAY)
#===ROI===
#oroi=o_gray[378:1050,1570:2215]
oroi_c=o.copy()
oroi_c=cv.rectangle(oroi_c,(2215,1050),(1570,378),(255,0,0),5)
#===DISPLAY ROI=======
cv.namedWindow("Ground Truth", cv.WINDOW_NORMAL)
cv.resizeWindow("Ground Truth", 800, 600)
cv.imshow("Ground Truth", oroi_c)
cv.waitKey(0)
cv.destroyAllWindows()
#=======================================================================================================

for _, row in pm.iterrows():
    f=row['ImageFile']
    i = cv.imread('/home/priyasha/new data/roi_preset_combined_train/' + f)
    i_gray=cv.cvtColor(i,cv.COLOR_BGR2GRAY)
    dz=f[:-4]+'_d.png'
    d=cv.imread('/home/priyasha/new data/roi_preset_combined_train_dehazed/'+dz)

    if d is None:
        print(f"Missing: {dz}")
        continue

    d_gray=cv.cvtColor(d,cv.COLOR_BGR2GRAY)
    strel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    e=cv.erode(d_gray,strel)
    #ROI_1==============================================================================================
    #====================ROI=============================
    #groi=i_gray[466:1155,1684:2268]
    groi=i_gray[378:1050,1570:2215]

    droi=d_gray[378:1050,1570:2215]
    eroi=e[378:1050,1570:2215]
    #=====================SSIM==================

    gr_ss=structural_similarity(groi,droi)
    er_ss=structural_similarity(groi,eroi)
   # or_ss=structural_similarity(groi,oroi)

    #=============STORE ROI==================================================

    roi_1.append(gr_ss)
    eroi_1.append(er_ss)
    #oroi_1.append(or_ss)
    #==================================================================================================
    #ROI_2=============================================================================================
    #==================ROI=================================
    groi1=i_gray[]
    droi1=d_gray[]
    eroi1=e[]
    #===================================================
    gr1_ss=structural_similarity(groi1,droi1)

belur_pm = pd.read_csv('/home/priyasha/new data/frb(train).csv')

belur_pm['SSIM_ROI']=roi_1
belur_pm['ERODE_SSIM_ROI']=eroi_1
#belur_pm['O_SSIM_ROI']=oroi_1

belur_pm.to_csv('/home/priyasha/new data/frb(train).csv',index=False)

