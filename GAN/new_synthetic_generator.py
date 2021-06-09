# `python new_synthetic_generator.py --i "image.fits" --o "final" --n 2`
"""
It is a modification of the synthetic_generator.py file that was created in previous project. You can find the original file on : 
https://github.com/YBouquet/detectsat
"""
from utils.mosaic import *
from utils.lines import *
from utils.img_processing import morphological_reconstruction
import utils.prologue as prologue
import numpy as np
import cv2
import os

import operator
import random
import gc

DATAPATH = "trainset/"

def saturated_stars(unscaled_img):
    """
    Get a mask with all saturated light blob by thresholding and morphological reconstruction
    """
    sigma = np.std(unscaled_img.flatten())
    indices = np.argwhere(unscaled_img > 3*sigma)
    mask = np.zeros(unscaled_img.shape).astype(np.uint8)
    mask[indices[:,0], indices[:,1]] = 1
    mask_1 = np.zeros(unscaled_img.shape).astype(np.uint8)
    indices = np.argwhere( unscaled_img > np.mean(unscaled_img))
    mask_1[indices[:,0], indices[:,1]] = 1
    final_mask = morphological_reconstruction(mask, mask_1, 2)
    return final_mask #, boxes

def blurry_effect(image, length) : 
    blurry = np.zeros(shape=(length,length))
    treshold = np.random.uniform(low = 0.2, high = 0.5)
    for i2 in range(length) : 
        for j2 in range(length) : 
            mean = int(length/2 - 1)
            x = -mean + j2 
            y = -mean + i2 
            dist = np.sqrt(x**2 + y**2) / (np.sqrt(2*mean**2))
            if image[i2,j2,0] != 0 : 
                if dist > treshold : 
                    new_value = image[i2,j2,0] - ((255-100)*dist)
                else :
                                                    # keep a core without decreasing
                    new_value = image[i2,j2,0] 
            else : 
                new_value = 0
        
            blurry[i2,j2] = new_value  
    return blurry 



def main(args, seed = 123456789):
    raw_image, unscaled_img = get_raw_image(args.i)#"OMEGA.2020-01-29T03_51_46.345_fullfield_binned.fits")
    crops_addresses = get_blocks_addresses(raw_image)

    x_train = []
    y_train = []
    has_satellite = []
    tmp_mask = []

    random.seed(seed)
    #4,6
    for i in range(4):
        for j in range(6):
            x_ = list(crops_addresses.keys())[i]
            crop = get_block(raw_image, x_, crops_addresses[x_][j])
            unscaled_crop = get_block(unscaled_img, x_, crops_addresses[x_][j])

            final_mask = saturated_stars(unscaled_crop) # detect saturated light blob in the image
            mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            final_mask=cv2.dilate(final_mask,mask_dil, iterations=1)
            h,w = crop.shape
            subh, subw = 256, 256
            for alpha in range(0,h, subh):
                for beta in range(0,w, subw):
                    if (alpha + subh) <= h and (beta+subw) <= w :
                        for k in range(args.n): # number of samples generated from a single 64x64 patch
                            subcrop = crop[alpha:alpha+subh, beta:beta+subw]
                            x_mirror = random.randint(0,1)
                            y_mirror = random.randint(0,1)
                            if x_mirror == 1:
                                subcrop = subcrop[::-1]
                            if y_mirror == 1:
                                subcrop = subcrop[:,::-1]
                            refcrop = subcrop.copy()
                            subdilation = final_mask[alpha:alpha+subh, beta:beta+subw]
                            star_indices = np.argwhere(subdilation == 1)
                            replacement_value = np.median(subcrop)
                            #subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value # remove outlier pixels
                            max_ = np.max(subcrop)
                            min_ = np.min(subcrop)
                            if max_ != min_ : # white patch
                                subcrop = (subcrop - min_) / (max_ - min_) # rescaling
                                y_true = np.zeros(subcrop.shape)
                                mask = np.full(subcrop.shape, 0.)
                                decision = random.random()
                                hs = 0
                                if decision < 0.5:
                                    hs = 1
                                    # STREAK PARAMETRIZATION
                                    s_length = random.randint(15,80) # chose the length of the streak
                                    s_width = random.randint(3,5) # chose the width of the streak
                                    theta = random.randint(0,179) * math.pi / 180. # chose the direction of the streak (in radian)
                                    x_where, y_where = random.randint(0, subh-s_length-1), random.randint(0,subw-s_length-1) # chose the position of the streak

                                    # DRAWING
                                    sat_line = np.zeros((s_length,s_length,3)).astype(np.uint8)
                                    _,p1,p2 = get_points(0,theta)
                                    sat_line = cv2.line(sat_line, tuple(map(operator.add, p1,(int(s_length/2),int(s_length/2)))),tuple(map(operator.add, p2,(int(s_length/2),int(s_length/2)))), (255,255,255), s_width)

                                    # APPLY THE STREAK IN THE PATCH
                                    h_sat,w_sat,_ = sat_line.shape
                                    cx,cy = float(h_sat//2), float(w_sat//2)
                                    r = min(cx,cy)
                                    for a in range(h_sat):
                                        for b in range(w_sat):
                                            if math.sqrt((a - cx)**2 + (b-cy)**2) > r:
                                                sat_line[a,b] = 0
                                    
                                    blurry_random = np.random.uniform(low = 0, high = 1 )
                                    if blurry_random <= 0.5 : #around 50% of blurry effect
                                        blur_sat_line = blurry_effect(sat_line, s_length)    
                                    else : 
                                        blur_sat_line = sat_line[:,:,0]
                                    
                                    #final_synth = cv2.GaussianBlur(sat_line/255,(s_width*2-1,s_width*2-1),0)
                                    final_synth = cv2.GaussianBlur(blur_sat_line/255,(s_width*2-1,s_width*2-1),0)
                                    #final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-1,s_width*2-1),0)
                                    alpha_trans = random.randint(65,95)/100. # opacity of the streak
                                    final_synth = (final_synth / np.max(final_synth))
                                    tmp_mask.append(final_synth)
                                    mask[x_where:x_where+s_length, y_where:y_where+s_length] = final_synth
                                    indices = np.argwhere(mask > 0.)
                                    for subx, suby in indices :
                                        subcrop[subx,suby] = max(alpha_trans * mask[subx,suby] + (1-alpha_trans) * subcrop[subx,suby], subcrop[subx,suby])
                                    #y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line / 255).astype(int)
                                    y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line[:,:,0] / 255).astype(int)
                                    del sat_line
                                    del blur_sat_line
                                    gc.collect()
                                subcrop = subcrop * (max_ - min_) + min_
                                subcrop[star_indices[:,0], star_indices[:,1]] = refcrop[star_indices[:,0], star_indices[:,1]] # put the    blobs in the image back
                                sub_max = np.max(subcrop)
                                sub_min = np.min(subcrop)
                                subcrop = (subcrop - sub_min) / (sub_max - sub_min)

                                # SAVING THE SAMPLES
                                has_satellite.append(hs)
                                x_train.append([subcrop])
                                y_train.append(y_true)
            del y_true
            del mask
            del refcrop
            del subcrop
            gc.collect()
    del crop
    del unscaled_crop
    del final_mask
    gc.collect()

    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    np.save(DATAPATH + args.o + "_samples.npy", np.array(x_train))
    np.save(DATAPATH + args.o + "_targets.npy", np.array(y_train))
    np.save(DATAPATH + args.o + "_patch_targets.npy", np.array(has_satellite))


if __name__ == '__main__':
    main(prologue.get_args())
