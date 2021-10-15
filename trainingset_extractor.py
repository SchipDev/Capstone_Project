import sys
import pandas as pd
import numpy as np

# Global Variables
SET_SIZE = int(sys.argv[1])    # Number of rows to extract into subset

READ_PATH = "../../../../projects/cmda_capstone_2021_ti/data/Data/"   # Path of csv file to read from
WRITE_PATH = "../../../../projects/cmda_capstone_2021_ti/data/training_sets/"    # Path of directory where output training csv file is to be saved

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