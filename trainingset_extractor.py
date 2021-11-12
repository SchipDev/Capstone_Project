import sys
import pandas as pd
import numpy as np 
import os
import shutil as sh

# RGB chips have an alpha channel

# Global Variables
SET_SIZE = int(sys.argv[1])    # Number of rows to extract into subset

READ_PATH = "/projects/cmda_capstone_2021_ti/data/data_summary_final_summary.csv"   # Path of csv file to read from
WRITE_PATH = "/projects/cmda_capstone_2021_ti/data/training_sets/"    # Path of directory where output training csv file is to be saved

NCHIP_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/NativeChips/"
RGBCHIP_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/ColorChips/"
FIVE_MASK_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/05masks/"

NCHIP_SUBPATH = "NativeChips"
RGBCHIP_SUBPATH = "ColorChips"
FIVE_MASK_SUBPATH = "05masks"


summary = pd.read_csv(READ_PATH)
print(summary.head(3))

cols = ['05min_Lightning_Count', '15min_Lightning_Count','30min_Lightning_Count']
summary[cols] = summary[cols].apply(pd.to_numeric, errors='coerce')

# Sort training set by number of lightning events in descending order
summary.sort_values(by='05min_Lightning_Count', ascending=False)

# Write training set to file
training_set = summary.head(SET_SIZE)
training_set.to_csv(WRITE_PATH + "trainingset_descending_" + str(SET_SIZE) + ".csv", index=False)
print(WRITE_PATH + "trainingset_descending_" + str(SET_SIZE) + ".csv" + "created with " + str(SET_SIZE) + "records!")

#------------------------------------------------
# Creating directory for training images
nchip_path = os.path.join(WRITE_PATH, NCHIP_SUBPATH)
rgbchip_path = os.path.join(WRITE_PATH, RGBCHIP_SUBPATH)
five_path = os.path.join(WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir = os.mkdir(nchip_path)
rgbchip_dir = os.mkdir(rgbchip_path)
fivemask_dir = os.mkdir(five_path)
print("All subfolders created succesfully")

#------------------------------------------------
#Copying files from main folders to trainingset subfolders

zipped = zip(training_set["Colorized_Chip_Name"], training_set["Native_Chip_Name"], training_set["05min_Mask_Name"])
record_loss = 0 # Records how many files are missing
for color, native, mask05 in zipped:
    if (color != "None") and (native != "None") and (mask05 != "None"):
        sh.copy(NCHIP_READ_PATH + native, nchip_path)
        sh.copy(RGBCHIP_READ_PATH + color, rgbchip_path)
        sh.copy(FIVE_MASK_READ_PATH + mask05, five_path)
    else:
        record_loss += 1

if record_loss == 0:
    print("All images copied to subfolder succesfully!")
else:
    print("Images copied to directories succesfully, " + str(record_loss) + " records were missing and could not be copied.")


print("Executed succesfully, go train that model stud!")