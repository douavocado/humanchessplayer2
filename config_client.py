#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:00:33 2020

@author: jx283
"""
import configparser

from pynput import mouse
from pynput.mouse import Button


CLICK_COUNT = 0
username = None
difficulty = None
TOP_LEFT = None
DX = None
DY = None
stockfish_path = None

def on_move(x, y):
    pass

def on_click(x, y, button, pressed):
    global CLICK_COUNT, TOP_LEFT, DX, DY
        
    if not pressed and button == Button.right:
        if CLICK_COUNT == 0:
            print('Registered right button click at X:{0}, Y:{1}'.format(x,y))
            print('Please right click ONCE the top right point of the top right square on the chessboard.')
            TOP_LEFT = (x, y)
        elif CLICK_COUNT == 1:
            print('Registered right button click at X:{0}, Y:{1}'.format(x,y))
            if x < TOP_LEFT[0]:
                print('Are you sure you clicked correctly? Please try again.')
                print('Please right click ONCE the top right point of the top right square on the chessboard.')
                return True
            print('Please right click ONCE the bottom left point of the bottom left square on the chessboard.')
            DX = x - TOP_LEFT[0]
        elif CLICK_COUNT == 2:
            print('Registered right button click at X:{0}, Y:{1}'.format(x,y))
            if y < TOP_LEFT[1]:
                print('Are you sure you clicked correctly? Please try again.')
                print('Please right click ONCE the bottom left point of the bottom left square on the chessboard.')
            print('Successful! Finished mouse calibration.')
            DY = y - TOP_LEFT[1]
            return False
        CLICK_COUNT += 1

def on_scroll(x, y, dx, dy):
    pass

config = configparser.ConfigParser()
config.read('config.ini')


if __name__ == '__main__':
    while True:
        print('Configuration for Lichess Client. What do you want to configure?')
        print('1. USERNAME')
        print('2. MOUSE POSITIONS')
        print('3. ENGINE PLAYING DIFFICULTY')
        print('4. PATH TO STOCKFISH BINARY')
        print('5. Save changes, quit.')
        while True:
            option = input('Option (1,2,3,4,5): ')
            if option not in ['1', '2', '3', '4', '5']:
                print('Unrecognized option, please try again.')
            else:
                break
        
        if option == '1':
            username = input('Please enter your new username (case sensitive): ')
        elif option == '2':
            print('Please visit lichess.org/tv and use the board there for calibration.')
            print('Please right click ONCE the top left point of the top left square on the chessboard.')
            # Collect events until released
            with mouse.Listener(
                    on_move=on_move,
                    on_click=on_click,
                    on_scroll=on_scroll) as listener:
                listener.join()
            
            # ...or, in a non-blocking fashion:
            listener = mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll)
            
            listener.start()
            print (TOP_LEFT, DX, DY)
            listener.stop()
        elif option == '3':
            print('The engine difficulty is how many \'human moves\' for stockfish to consider.')
            print('Setting a low difficulty (1-5) would cause many obvious moves to not be picked up by engine (Typically <1500 level on lichess).')
            print('Setting a high difficulty would mean the engine would play more and more like stockfish (Easy to detect cheating).')
            print('Recommended difficulty is between 6-10, perhaps a little higher for when in shadow mode.')
            while True:
                difficulty = input('Difficulty level (1-20): ')
                try:
                    difficulty = int(difficulty)
                    if difficulty < 21 and difficulty >= 1:
                        break
                    else:
                        print('Please select a difficulty level within the required bounds.')
                except:
                    print('Unrecognized difficulty level, please try again.')
        elif option == '4':
            stockfish_path = input('Please enter the FULL path to the stockfish binary: ')
        elif option == '5':
            # saving details to config file
            if username is not None:
                config['lichess.org']['username'] = username
            if difficulty is not None:
                config['DEFAULT']['difficulty'] = str(difficulty)
            if stockfish_path is not None:
                config['DEFAULT']['path'] = stockfish_path
            if TOP_LEFT is not None:
                config['DEFAULT']['start_x'] = str(TOP_LEFT[0])
                config['DEFAULT']['start_y'] = str(TOP_LEFT[1])
                config['DEFAULT']['step'] = str((DX + DY)/16)
            
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            print('Changes successful')
            break