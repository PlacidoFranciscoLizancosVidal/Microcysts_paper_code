import matplotlib.pyplot as plt
from util_storage import loadObj
import os, json
import numpy as np
from matplotlib.pyplot import figure

%matplotlib inline

target_folder = './Minicist_V2/'
epoch = '0'

f = figure(figsize=(8, 3.5), dpi=300)
ax = f.gca()
for window_size in target_folder:


epochpath = target_folder + epoch + '/'
for folder in os.listdir(epochpath):