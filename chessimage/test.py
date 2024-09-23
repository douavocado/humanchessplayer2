# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:54:05 2023

@author: xusem
"""

import cv2
import pyautogui
import numpy as np
import time

from scraping import get_fen_from_image, get_move_change

time.sleep(5)
start = time.time()
im = pyautogui.screenshot('my_screenshot.png', region=(595,242, 632, 632))
end = time.time()
print('screenshot taken in ', end - start)
start = time.time()
image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
print(get_move_change(image))
end = time.time()
print('Move change detected in', end - start)


    
        
        