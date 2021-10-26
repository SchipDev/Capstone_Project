import sys
import pandas as pd
import numpy as np 
import os
import shutil as sh

# Global Variables
SET_SIZE = int(sys.argv[1])    # Number of rows to extract into subset

READ_PATH = "/projects/cmda_capstone_2021_ti/data/data_summary_final_summary.csv"   # Path of csv file to read from
WRITE_PATH = "/projects/cmda_capstone_2021_ti/data/training_sets/"    # Path of directory where output training csv file is to be saved

NCHIP_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/NativeChips/"
FIVE_MASK_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/05masks/"

NCHIP_SUBPATH = "NativeChips"
FIVE_MASK_SUBPATH = "05masks"


summary = pd.read_csv(READ_PATH)
print(summary.head(3))

# Convert the lightning count columns to numeric values
for num in range(len(summary["05min_Lightning_Count"])):
    if summary["05min_Lightning_Count"][num] == "None":
        summary["05min_Lightning_Count"][num] = "0"

summary["05min_Lightning_Count"] = pd.to_numeric(summary["05min_Lightning_Count"])

for num in range(len(summary["15min_Lightning_Count"])):
    if summary["15min_Lightning_Count"][num] == "None":
        summary["15min_Lightning_Count"][num] = "0"

summary["15min_Lightning_Count"] = pd.to_numeric(summary["15min_Lightning_Count"])

for num in range(len(summary["30min_Lightning_Count"])):
    if summary["30min_Lightning_Count"][num] == "None":
        summary["30min_Lightning_Count"][num] = "0"

summary["30min_Lightning_Count"] = pd.to_numeric(summary["30min_Lightning_Count"])

# Sort training set by number of lightning events in descending order
summary.sort_values(by='05min_Lightning_Count', ascending=False)

# Write training set to file
training_set = summary.head(SET_SIZE)
training_set.to_csv(WRITE_PATH + "trainingset_descending_" + str(SET_SIZE) + ".csv", index=False)
print(WRITE_PATH + "trainingset_descending_" + str(SET_SIZE) + ".csv" + "created with " + str(SET_SIZE) + "records!")

#------------------------------------------------
# Creating directory for training images
nchip_path = os.path.join(WRITE_PATH, NCHIP_SUBPATH)
five_path = os.path.join(WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir = os.mkdir(nchip_path)
fivemask_dir = os.mkdir(five_path)
print("All subfolders created succesfully")

#------------------------------------------------
#Copying files from main folders to trainingset subfolders

# Copying native chips
for nchip in training_set["Native_Chip_Name"]:
    sh.copy(NCHIP_READ_PATH + nchip, nchip_path)

# Copying 05 masks
for fivemask in training_set["05min_Mask_Name"]:
    sh.copy(FIVE_MASK_READ_PATH + fivemask, five_path)

print("All images copied to subfolder succesfully!")


print("Executed succesfully, go train that model stud!")