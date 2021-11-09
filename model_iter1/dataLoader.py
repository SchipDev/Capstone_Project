import pandas as pd
import numpy as np
from matplotlib import image
import sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

# File loads in image data given from a csv file containing img paths into a numpy array and returns it.

IMG_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/"


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

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
        _05mask_arr = _05mask_arr.clip(max=1)
        #_05mask_arr = _05mask_arr.reshape([-1,400, 400,1])

        segmented = blockshaped(_05mask_arr.reshape(400,400),10,10)
        for i in range(len(segmented)):
            if(np.max(segmented[i]) != 0):
                segmented[i,:,:]=1
        segmented = unblockshaped(segmented,400,400)
        segmented = segmented.reshape(400,400,1)

        _05mask_array.append(np.asarray(segmented))
        it2+=1

    

    
    return np.asarray(nchip_arr), np.asarray(_05mask_array)

    
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        # img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



    

