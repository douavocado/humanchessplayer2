#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:27:17 2020

@author: jx283
"""
import argparse
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

parser = argparse.ArgumentParser(description='Establish connection to lichess.org account.')
parser.add_argument("-s", "--shadow", help="Run in shadow mode, where the mouse hovers over it's recommended move approximately 1 second after position found",
                        action="store_true")
parser.add_argument("-n", "--nolog", help="Disable log output into /Engine_logs/",
                        action="store_true")
args = parser.parse_args()

# from lichess_premove_continuous_client import GameFinder 
from no_scrape_lichess_client import GameFinder   

# clearing memory
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

finder = GameFinder(shadow_mode=args.shadow, log= not args.nolog)
finder.run(games=10)