import cv2
import numpy as np
import math
import pandas as pd
from scipy import stats
import os
####################
inputFiles='/media/seagate/vmeo/detectionDataSet/general/F/img'
#inputFiles='/media/seagate/vmeo/detectionDataSet/general/T'
resultFile='./dataset/train/fake/Deepfakes/000_162'
#resultFile='./dataset/train/real/000'
FAKEimg=True



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
                    minY=np.min(y_nonzero)
                    maxY=np.max(y_nonzero)
                    minx=np.min(x_nonzero)
                    maxx=np.max(x_nonzero)
                    if np.max(y_nonzero)-np.min(y_nonzero)!=128:
                        taille=(np.max(y_nonzero)-np.min(y_nonzero))
                        if taille<128:
                                if (np.max(y_nonzero)+(128-taille))>255:
                                   minY=minY-(128-taille) 
                                else:
                                    maxY=maxY+(128-taille)
                                    
                        else:        
                           maxY=minY+128
                    if np.max(x_nonzero)-np.min(x_nonzero)!=128: 
                         taille=(np.max(x_nonzero)-np.min(x_nonzero))
                         if taille<128:
                                 if (np.max(x_nonzero)+(128-taille))>255:
                                    minx=minx-(128-taille) 
                                 else:
                                     maxx=maxx+(128-taille)
                                     
                         else:        
                            maxx=minx+128                       
                            
                    
                    return image[minY:maxY, minx:maxx]
                image=crop(image)
            
         
          else:
              image=cv2.imread(fp)
              image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
              dimensionn=image.shape
              image = image[int(dimensionn[0]/2)-64:int(dimensionn[0]/2)+64, int(dimensionn[1]/2)-64:int(dimensionn[1]/2)+64]


          name=fp.split('/')[-1]
          
          filename=os.path.join(resultFile,name)
        
          cv2.imwrite(filename, image)

            
            
