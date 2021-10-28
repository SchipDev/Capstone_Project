import pandas as pd
import numpy as np
from matplotlib import image
import sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

# File loads in image data given from a csv file containing img paths into a numpy array and returns it.

IMG_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/"

def load_data(path):
    data = pd.read_csv(path)
    # print(path + " loaded, " + len(data.index) + " records detected.")
    nchip_arr = []
    _05mask_array = []
    it1 = 0
    it2 = 0
    for im in data["Native_Chip_Name"]:
        img_PIL = load_img(IMG_READ_PATH + "NativeChips/" + im, color_mode = "grayscale")
        img_array = img_to_array(img_PIL)
        #img_array = img_array.reshape([-1,400, 400,1])
        nchip_arr.append(np.asarray(img_array))
        it1+= 1

    for mask in data["05min_Mask_Name"]:
        _05mask = load_img(IMG_READ_PATH + "05masks/" + mask, color_mode="grayscale")
        _05mask_arr = img_to_array(_05mask)
        _05mask_arr = _05mask_arr.clip(max=5)
        #_05mask_arr = _05mask_arr.reshape([-1,400, 400,1])
        _05mask_array.append(np.asarray(_05mask_arr))
        it2+=1

    

    
    return np.asarray(nchip_arr), np.asarray(_05mask_array)

    



