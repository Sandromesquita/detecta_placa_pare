# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:39:47 2021

@author: sandr
"""

import cv2 
from matplotlib import pyplot as plt
import time

placas = ["image.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg",
          "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg", "gatos/image10.jpg"]

for i in range(len(placas)):
    img = cv2.imread(placas[i]) 
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    stop_data = cv2.CascadeClassifier('stop_data.xml') 
    found = stop_data.detectMultiScale(img_gray, minSize =(20, 20)) 
    amount_found = len(found) 
      
    if amount_found != 0:   
        for (x, y, width, height) in found:        
            cv2.rectangle(img_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)  
            
    plt.subplot(1, 1, 1) 
    plt.imshow(img_rgb) 
    plt.show()
    time.sleep(1)