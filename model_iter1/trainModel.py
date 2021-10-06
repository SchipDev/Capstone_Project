from model import *
import pandas as pd
import numpy as np
from matplotlib import image
from matplotlib import pyplot
from dataLoader import *

# Load Training/Testing Data
native_chips = load_data("../../data/training_sets/trainingset_descending_40.py")

# Construct model
model = unet()


# Fit Model
preds = model.predict(native_chips)
print(preds)

# Save Model