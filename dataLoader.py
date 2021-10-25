import pandas as pd
import numpy as np
from matplotlib import image
import sys
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

# File loads in image data given from a csv file containing img paths into a numpy array and returns it.

IMG_READ_PATH = "../../../../../projects/cmda_capstone_2021_ti/data/Data/"

def load_data(path):
    data = pd.read_csv(path)
    #print(path + " loaded, " + len(data.index) + " records detected.")
    nchip_arr = []
    _05mask_array = []
    for im in data["Native_Chip_Name"]:
        img_PIL = load_img(IMG_READ_PATH + "NativeChips/" + im, color_mode = "grayscale")
        img_array = img_to_array(img_PIL)
        nchip_arr.append(img_array)

    for mask in data["05min_Lightning_Count"]:
        _05mask = load_img(IMG_READ_PATH + "05masks" + mask, color_mode="grayscale")
        _05mask_arr = img_to_array(05mask)
        _05mask_arr = _05mask_arr.clip(max=5)
        _05mask_array.append(_05mask_arr)

    


    
    return nchip_arr, _05mask_array



