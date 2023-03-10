#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2019

@author: Jiatai Han
"""
import cv2
import numpy as np
from ViolaJonesFaceDetector import FaceDetector
from utils import nonMaximalSupression,getImages

# Change your photo here
testimage = './read-world_photos/testimage5.jpg'


def getFaceLocations(gray, img):

    print('Start recognition...')

    imgcp = img.copy()
    imgcp2 = img.copy()
    
    colssize = int(gray.shape[1]/5)
    step = int(colssize/4)
    rowssize = colssize
                                    
    clf = FaceDetector.Load('Classifier')
            
    locations = []
            
    while rowssize<(len(gray)-2):

        for r in range(0,gray.shape[0] - rowssize, step):    
            
            for c in range(0,gray.shape[1] - colssize, step):
                        
                window = gray[r:r+rowssize, c:c+colssize]
                windowColor = imgcp2[r:r+rowssize, c:c+colssize]

                window=cv2.resize(window,dsize=(19,19))
                #windowColor=cv2.resize(windowColor,dsize=(19,19))

                prediction = clf.classify(window)

                img = imgcp.copy()
                

                for i in clf.alphas:
                    top_index = clf.alphas.index(i)
                    clf.getBoxImageShow(clf.infos[top_index][0], len(clf.classifiers), windowColor, int(colssize/19), img, c, r, colssize, rowssize)

                    cv2.rectangle(img, (c-1, r-1), (c+colssize+1, r+rowssize+1), (0,0,255), 2)

                cv2.imshow("window", img)
                #cv2.imshow("w2", window)
                cv2.waitKey(1)
                        
                if prediction == 1:
                    cv2.rectangle(imgcp, (c, r), (c+colssize ,r+rowssize), (0,255,0), 1)
                    locations.append([r,c,r+rowssize,c+colssize])
                

                
                    
        colssize+=step
        
        rowssize+=step
    
    return locations
    
           
        
imgtest = cv2.imread(testimage)                    
gray = cv2.cvtColor(imgtest,cv2.COLOR_RGB2GRAY)
print('Image', testimage, ' is loaded')

# Normalize
mean = gray.mean()
std = gray.std()
std = gray.std()

#locations = [[1,2,3,4],[1,2,3,4]]
locations = getFaceLocations(gray, imgtest)
            
if locations:
    locations = nonMaximalSupression(np.array(locations), 0)
    print('Face(s) recognized at', locations)
    for location in locations:
        cv2.rectangle(imgtest, (location[0], location[1]), (location[2], location[3]), (255,0,0), 2)
    #cv2.imwrite("result.png",imgtest)
    cv2.imshow("window", imgtest)
    k = cv2.waitKey(0) # 0==wait forever
else:
    print('Face not recognized, please change to another photo.')


    
    
    
    
        
    
    
    

    
