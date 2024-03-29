#-*-coding:utf-8-*-
#by Yifan Sun
import cv2
import numpy as np
import math
import pandas as pd
from scipy import stats
import os
####################
inputFiles='/media/seagate/vmeo/detectionDataSet/general/F/img'
#inputFiles='/media/seagate/vmeo/detectionDataSet/general/T'
resultFile='./feature-general-Fake.csv'
FAKEimg=True
###################

#[Spatial Domain] Image Colorfulness Index (CFI)
def calcuCFI(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)


#[Color Histogram] Color Moments (CM1_R,CM2_R,CM3_R,CM1_G,CM2_G,CM3_G,CM1_B,CM2_B,CM3_B)
def calcuCM(image):
    (B, G, R) = cv2.split(image.astype("float"))
    #img= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    CM1_R=R.mean()
    CM2_R=R.std()
    mid_R = np.mean(((R - R.mean()) ** 3))
    CM3_R= np.sign(mid_R) * abs(mid_R) ** (1/3)
    CM1_G = G.mean()
    CM2_G = G.std()
    mid_G = np.mean(((G - G.mean()) ** 3))
    CM3_G = np.sign(mid_G) * abs(mid_G) ** (1 / 3)
    CM1_B = B.mean()
    CM2_B = B.std()
    mid_B = np.mean(((B - B.mean()) ** 3))
    CM3_B = np.sign(mid_B) * abs(mid_B) ** (1 / 3)

    return CM1_R,CM2_R,CM3_R,CM1_G,CM2_G,CM3_G,CM1_B,CM2_B,CM3_B

#[Color Histogram] Metrics for the grayscale histogram (MEAN, STD, SKEW, KURT, IET)
def calcuHist(image):
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = pd.Series(image2.flatten())
    probs = pd.Series(image2.flatten()).value_counts() / len(image2.flatten())
    IET = stats.entropy(probs, base=2)
    MEAN=image2.mean()
    STD=image2.std()
    SKEW=img.skew()
    KURT=img.kurt()
    return MEAN,STD,SKEW,KURT,IET

#[Color Histogram] Image quility metrics (BIQ, TIQ, LIQ)
def calcuIQM(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img=img/255.0
    x,y=img.shape
    resBre=(img[2:x,:]-img[0:x-2,:])**2
    resLap = cv2.Laplacian(img, cv2.CV_64F)
    resSobx = cv2.Sobel(img, cv2.CV_64F,1,0)
    resSoby = cv2.Sobel(img, cv2.CV_64F,0,1)
    resSob =(resSobx**2+resSoby**2)**0.5
    BIQ=resBre.sum()
    TIQ=resSob.sum()
    LIQ=(np.abs(resLap)).sum()

    return BIQ,TIQ,LIQ

#[Spatial Domain] Texture Metrics (ASM,CON,ENT,IDM)
def calcuGLCM(img,d_x,d_y):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    srcdata=img.copy()
    p = [[0.0 for i in range(256)] for j in range(256)]
    (height, width) = img.shape
    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            p[rows][cols] += 1.0
    for i in range(256):
        for j in range(256):
            p[i][j] /= float(height * width)
    CON = 0.0
    ENT = 0.0
    ASM = 0.0
    IDM = 0.0
    for i in range(256):
        for j in range(256):
            CON += (i - j) * (i - j) * p[i][j]
            ASM += p[i][j] * p[i][j]
            IDM += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                ENT -= p[i][j] * math.log(p[i][j])
    return ASM,CON,ENT,IDM

#[Frequency Domain] Metrics for frequency domain (FASM, FCON, FENT, FIDM)
def calcufGLCM(img,d_x,d_y):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fimg= np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    srcdata = (fimg - fimg.min()) / (fimg.max() - fimg.min()) * 255
    srcdata = srcdata.astype(np.int16)
    p = [[0.0 for i in range(256)] for j in range(256)]
    (height, width) = fimg.shape
    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            p[rows][cols] += 1.0
    for i in range(256):
        for j in range(256):
            p[i][j] /= float(height * width)
    FCON = 0.0
    FENT = 0.0
    FASM = 0.0
    FIDM = 0.0
    for i in range(256):
        for j in range(256):
            FCON += (i - j) * (i - j) * p[i][j]
            FASM += p[i][j] * p[i][j]
            FIDM += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                FENT -= p[i][j] * math.log(p[i][j])
    return FASM, FCON, FENT, FIDM

if __name__ == '__main__':
    fps = list()

    for root, dirs, files in os.walk(inputFiles):
        for file in files:
            fp = os.path.join(root, file)
            fps.append(fp)
    n=len(fps)
    print(n)
    ID=0
    result=[]
    for fp in fps:
        if FAKEimg:
            mask = fp.split('/')
            mask[-2]='mask'
            mask="/".join(mask)
#            print(mask)
            mask=cv2.imread(mask)
            image=cv2.imread(fp)
            image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
            #print(image.shape)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #print(mask.shape)
            image = cv2.bitwise_and(image,image,mask = mask)
            def crop(image):
                y_nonzero, x_nonzero, _ = np.nonzero(image)
                return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
            image=crop(image)

        else:
            image=cv2.imread(fp)
            image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
            dimensionn=image.shape
            image = image[int(dimensionn[0]/2)-64:int(dimensionn[0]/2)+64, int(dimensionn[1]/2)-64:int(dimensionn[1]/2)+64]
        ID+=1
        FP=fp
        dirnames = fp.split('/')
        PID=dirnames[len(dirnames)-2]+dirnames[len(dirnames)-1].split('.')[0]
        dirnames=fp.replace('\n', '').replace('\r', '').split('.')
        if FAKEimg:
            isFake=1
        else:
            isFake=0
        #Feature calculation
        MEAN, STD, SKEW, KURT, IET=calcuHist(image)
        CM1_R, CM2_R, CM3_R, CM1_G, CM2_G, CM3_G, CM1_B, CM2_B, CM3_B=calcuCM(image)
        CFI=calcuCFI(image)
        BIQ, TIQ, LIQ=calcuIQM(image)
        ASM, CON, ENT, IDM=calcuGLCM(image,1,1)
        FASM, FCON, FENT, FIDM=calcufGLCM(image,1,1)
        re=[ID,PID,FP,isFake,MEAN, STD, SKEW, KURT, IET,CM1_R, CM2_R, CM3_R, CM1_G, CM2_G, CM3_G, CM1_B, CM2_B, CM3_B,CFI,BIQ, TIQ, LIQ,ASM, CON, ENT, IDM,FASM, FCON, FENT, FIDM]
        result.append(re)
        print(str(ID)+'/'+str(len(fps)),FP)



    cols=['ID','PID','FP','isFake','MEAN','STD','SKEW','KURT','IET','CM1_R','CM2_R','CM3_R','CM1_G','CM2_G','CM3_G','CM1_B','CM2_B','CM3_B','CFI','BIQ','TIQ','LIQ','ASM','CON','ENT','IDM','FASM','FCON','FENT','FIDM']
    rdf=pd.DataFrame(result,columns=cols)
    rdf.to_csv(resultFile,index=False)
