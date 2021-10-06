import pandas as pd
import numpy as np
from matplotlib import image
import sys

# File loads in image data given from a csv file containing img paths into a numpy array and returns it.

def load_data(path):
    data = pd.read_csv(path)
    print(path + " loaded, " + len(data.index) + " records detected.")
    arr = np.array()
    for im in data["Native_Chip_Name"]:
        im_data = image.imread(im)
        arr.append(im_data)
    
    return arr

