from model import *
import pandas as pd
import numpy as np
from matplotlib import image, interactive
from matplotlib import pyplot
from dataLoader import *
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img
import os
import shutil as sh
import matplotlib.pyplot as plt

#os environ gpu
# Load Training/Testing Data
# native_chips, _05masks = load_data("/projects/cmda_capstone_2021_ti/data/training_sets/trainingset_descending_40.csv")

data_gen_args = dict(rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'/projects/cmda_capstone_2021_ti/data/training_sets','NativeChips','05masks',data_gen_args,save_to_dir = None)
print(myGene)
# Construct model
model = unet()


# Fit Model

# history = model.fit(native_chips, _05masks, epochs=140)
model.fit(myGene, epochs=140)
# os.mkdir("model_iter1/model_accuracy")
# plt.plot(history.history["accuracy"])
# # plt.plot(history.history["val_accuracy"])
# plt.title("Modle Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# #plt.legend(["Train", "Test"], loc='upper_left')
# plt.savefig("model_iter1/model_accuracy/Accuracy_Plot.png")

# preds = model.predict(native_chips[0].reshape([-1,400, 400,1]))
# pred_img = preds[0]
# true_chip = native_chips[0]
# true_mask = _05masks[0]
# save_img("model_iter1/model_accuracy/trained_model_sample_pred.png", pred_img.clip(max=1))
# save_img("model_iter1/model_accuracy/Predicted_native_chip.png", true_chip)
# save_img("model_iter1/model_accuracy/True_Mask.png", true_mask)

# # Save Model
# model.save("model_iter1/saved_model")
