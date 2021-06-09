import numpy as np 
import matplotlib.pyplot as plt
from utils.mosaic import scale_image, get_raw_image, get_blocks_addresses, get_block
import PIL
from PIL import Image
def real_image(file) : 
    """ This function returns the 32 blocks of the full mosaic of the given file """
    raw_image,unscaled_img = get_raw_image(file)
    crops_addresses = get_blocks_addresses(raw_image)

    x_values = list(crops_addresses.keys())
    y_values = list(crops_addresses[x_values[0]]) # same y_values for all 

    Blocs = []
    for i in x_values : 
        for j in y_values : 
            crop = get_block(raw_image, i, j)
            crop = ((crop - np.min(crop)) / (np.max(crop) -  np.min(crop)) * 255).astype(int).astype(float)
            crop = (crop - np.min(crop)) / (np.max(crop) -  np.min(crop))
            Blocs.append(crop)
            

    return Blocs

def Metrics(pred, true) :
    TP = 0; TN = 0 ;
    FP = 0; FN = 0; 
    for i in range(256) : 
        for j in range(256) : 
            if pred[i,j] == 1 : 
                if true[i,j] == 1 : 
                    TP +=1 
                else : 
                    FP +=1
            else : 
                if true[i,j] == 0:
                    TN +=1
                else : 
                    FN +=1
    
    Dice = 2 * TP / (2*TP + FP + FN) if TP != 0 else 0 
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall=TP/(TP+FN) if TP + FN != 0 else 0
    specificity = TN/(TN+FP) if TN + FP != 0 else 0 
    precision = TP / (TP + FP) if TP + FP != 0 else 0 
    NPV = TN / (TN + FN) if TN + FN != 0 else 0 
  
    return Dice, accuracy, recall, specificity, precision, NPV 

def shuffle_in_unison(a, b, seed):
    np.random.seed(seed)
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b