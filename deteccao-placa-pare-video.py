# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:09:57 2021

@author: sandr
"""

import cv2 
import numpy as np

# CORES DAS CLASSES
COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]
cap = cv2.VideoCapture('placas3.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    fgmask = fgbg.apply(gray)
    #cv2.imshow("fgmask", fgmask)
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow("th", th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 2)
    #cv2.imshow("opening", opening)
    dilation = cv2.dilate(opening,kernel,iterations = 8)
    #cv2.imshow("dilation", dilation)
    #closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 8)   
    stop_data = cv2.CascadeClassifier('stop_data.xml') 
    found = stop_data.detectMultiScale(frame, minSize =(20, 20)) 
    amount_found = len(found) 
      
    if amount_found != 0:   
        for (x, y, width, height) in found:        
            cv2.rectangle(frame, (x, y), (x + height, y + width), (0, 255, 0), 5) 
            
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    