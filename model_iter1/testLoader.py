from dataLoader import *
import numpy as np

native_chips, _05masks = load_data("/projects/cmda_capstone_2021_ti/data/training_sets/trainingset_descending_40.csv")
print(np.array(native_chips).shape)
