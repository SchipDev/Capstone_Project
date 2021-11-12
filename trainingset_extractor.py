from posixpath import split
import sys
import pandas as pd
import numpy as np 
import os
import shutil as sh

# RGB chips have an alpha channel

# Global Variables
SET_SIZE = int(sys.argv[1])    # Number of rows to extract into subset
VAL_SET_SIZE = 201

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
# Creating Train and validation sub-folders
TRAIN_WRITE_PATH = os.path.join(WRITE_PATH, "Train/")
VAL_WRITE_PATH = os.path.join(WRITE_PATH, "Val/")
train_dir = os.mkdir(TRAIN_WRITE_PATH)
val_dir = os.mkdir(VAL_WRITE_PATH)

# Creating directory for training images
nchip_path_t = os.path.join(TRAIN_WRITE_PATH, NCHIP_SUBPATH)
rgbchip_path_t = os.path.join(TRAIN_WRITE_PATH, RGBCHIP_SUBPATH)
five_path_t = os.path.join(TRAIN_WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir_t = os.mkdir(nchip_path_t)
rgbchip_dir_t = os.mkdir(rgbchip_path_t)
fivemask_dir_t = os.mkdir(five_path_t)

nchip_path_v = os.path.join(VAL_WRITE_PATH, NCHIP_SUBPATH)
rgbchip_path_v = os.path.join(VAL_WRITE_PATH, RGBCHIP_SUBPATH)
five_path_v = os.path.join(VAL_WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir_v = os.mkdir(nchip_path_v)
rgbchip_dir_v = os.mkdir(rgbchip_path_v)
fivemask_dir_v = os.mkdir(five_path_v)

print("All subfolders created succesfully")

#------------------------------------------------
#Copying files from main folders to trainingset subfolders

zipped = zip(training_set["Colorized_Chip_Name"], training_set["Native_Chip_Name"], training_set["05min_Mask_Name"])
record_loss = 0 # Records how many files are missing
split_ctr = 0
for color, native, mask05 in zipped:
    if (color != "None") and (native != "None") and (mask05 != "None"):
        if split_ctr >= SET_SIZE-VAL_SET_SIZE:
            sh.copy(NCHIP_READ_PATH + native, nchip_path_v)
            sh.copy(RGBCHIP_READ_PATH + color, rgbchip_path_v)
            sh.copy(FIVE_MASK_READ_PATH + mask05, five_path_v)
        else:
            sh.copy(NCHIP_READ_PATH + native, nchip_path_t)
            sh.copy(RGBCHIP_READ_PATH + color, rgbchip_path_t)
            sh.copy(FIVE_MASK_READ_PATH + mask05, five_path_t)
        split_ctr += 1
    else:
        record_loss += 1

if record_loss == 0:
    print("All images copied to subfolder succesfully!")
else:
    print("Images copied to directories succesfully, " + str(record_loss) + " records were missing and could not be copied.")


print("Executed succesfully, go train that model stud!")