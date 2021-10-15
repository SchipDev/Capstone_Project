from model import *
import pandas as pd
import numpy as np
from matplotlib import image
from matplotlib import pyplot
from dataLoader import *

# Load Training/Testing Data
native_chips, _05masks = load_data("../../../../../projects/cmda_capstone_2021_ti/data/training_sets/")

# Construct model
model = unet()


# Fit Model
preds = model.predict(native_chips)
print(preds[0])

# Save Model