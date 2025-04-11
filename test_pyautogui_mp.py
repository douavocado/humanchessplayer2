#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:51:37 2024

@author: james
"""
# import Xlib.threaded

# import pyautogui
# from multiprocessing import Process, Value
# import time

# def proc1():
#     for i in range(10):
           
#         current_x, current_y = pyautogui.position()
               
#         print("process1: {}, {}".format(current_x, current_y))
#         time.sleep(3)

# def proc2(mouse_locked):
#     for i in range(10):
           
#         current_x, current_y = pyautogui.position()
               
#         print("process2: {}, {}".format(current_x, current_y))
#         time.sleep(2)

# #mouse_locked = Value("i", 0)
# p1 = Process(target=proc1, )
# #p2 = Process(target=proc2, )
# p1.start()
# #p2.start()
# p1.join()
# #p2.join()
# p1.kill()
# #p2.kill()

import multiprocessing
import pyautogui
import time
import os

def worker(action, duration):
  """
  This function performs a specified action using pyautogui for a given duration.

  Args:
    action: The pyautogui function to execute (e.g., pyautogui.click, pyautogui.write).
    duration: The time in seconds to perform the action.
  """
  end_time = time.time() + duration
  while time.time() < end_time:
    action()
    # Optional: Add a small delay to reduce X server load
    time.sleep(0.1)

if __name__ == '__main__':
  # Define the actions you want to perform
  actions = [
      (pyautogui.click, 5),  # Click for 5 seconds
      (lambda: pyautogui.write('hello world', interval=0.25), 10)  # Write 'hello world' for 10 seconds
  ]

  # Create processes for each action
  processes = []
  for action, duration in actions:
    p = multiprocessing.Process(target=worker, args=(action, duration))
    processes.append(p)
    p.start()

  # Wait for all processes to finish
  for p in processes:
    p.join()

  print("All actions completed.")

  # Use xvfb-run to execute the script with a virtual display server
  #os.system("xvfb-run -a python test_pyautogui_mp.py")