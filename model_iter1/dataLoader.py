import pandas as pd
import numpy as np
from matplotlib import image
import sys
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

# File loads in image data given from a csv file containing img paths into a numpy array and returns it.

IMG_READ_PATH = "../../../../../projects/cmda_capstone_2021_ti/data/Data/NativeChips/"

def load_data(path):
    data = pd.read_csv(path)
    print(path + " loaded, " + len(data.index) + " records detected.")
    arr = np.array()
    for im in data["Native_Chip_Name"]:
        img_PIL = load_img(IMG_READ_PATH + im, color_mode = "grayscale")
        img_array = img_to_array(img_PIL)
        arr.append(img_array)
    
    return arr



