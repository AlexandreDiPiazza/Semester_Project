import numpy as np 
import os
import gc
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.utils import class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

np.random.seed(1234)
tf.random.set_seed(1234)
"""
This file trains 163 differents models and outputs the average results of the resulting models. 
It saves the result of each model in 'Fine_Tuning_Parameters.npy'

The main trains a model from model 1 checkpoints, and fine-tune on the real data with the parameters (epochs, learning-rate, etc)
on which it is called
"""
def main(I, EPOCH, LR, TR, DRO, X_vraie, Y_vraie) : 
    
   
    np.random.seed(1234)
    tf.random.set_seed(1234)

    def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def get_unet(input_img, n_filters = 32, dropout = 0.1, batchnorm = True):
    
    # Contracting Path
        c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)
    
        c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
    
        c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
    
        c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)
    
        c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

    callbacks = [
        EarlyStopping(patience=5, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=2, min_lr=1e-5, verbose=1),
    ]


    def jaccard_distance(y_true, y_pred):
    
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + 100.) / (sum_ - intersection + 100.)
        jd =  (1 - jac) * 100.
        return tf.reduce_mean(jd)


    import random
        
   

    def load_model(learning_rate, dropout,i):
        """
        Here we freeze some layers depending on the value of i, for fine-tuning the pre-trained model
        """
        im_height = 256
        im_width = 256
        input_img = Input((im_height, im_width, 1), name='img')
        model = get_unet(input_img, n_filters= 32, dropout=dropout, batchnorm=True)
        model.compile(optimizer=Adam(lr =learning_rate), loss="binary_crossentropy", metrics=[jaccard_distance])
        model.load_weights('model1')
    
        # depend on the values of i, block some layers
        index = 0
        for layer in model.layers : 
            index += 1
            
            if i == 1 : 
                if index == 46 or index == 49 : 
                    #unblock last 2 layers
                    layer.trainable = True
                else : 
                    layer.trainable = False
            elif i == 2 : 
                if index == 49 : 
                    # unblock last layers
                    layer.trainable = True
                else : 
                    layer.trainable = False
            
        return model 

    def build_k_indices(y, k_fold, seed):
      
        num_row = np.shape(y)[0]
        interval = int(num_row / k_fold)
        np.random.seed(seed)
        indices = np.random.permutation(num_row)
        k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
        return np.array(k_indices)
    def Dice_index(pred, true) :
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

    def results(model, x_test, y_test, tresh) :
        Dic = []; Acc = [] ; Rec = [] ; Spe = [] ; Pre = [] ; NPVs = [] ;
        for i in range(np.shape(x_test)[0]) : 
            prediction = model.predict(x_test[i,:,:].reshape(1,256,256,1)).reshape((256,256))
            prediction[prediction>tresh] = 1
            prediction[prediction<tresh] = 0
            Dice, accuracy, recall, specificity, precision, NPV = Dice_index(prediction, y_test[i,:,:])
            Dic.append(Dice);Acc.append(accuracy);Rec.append(recall);Pre.append(precision);Spe.append(specificity)
            NPVs.append(NPV)

        return sum(Dic)/len(Dic), sum(Acc)/len(Acc), sum(Rec)/len(Rec), sum(Spe)/len(Spe), sum(Pre)/len(Pre), sum(NPVs)/len(NPVs)

    # Perform K-Cross Validation for each model to have average results on validation set                                                                              
    k_fold = 4
    seed = 1
    k_indices = build_k_indices(X_vraie, k_fold, 1) 
    
    # All the metrics : Dice, Accuracy, Recall, Specificity, Precision, NPV
    Ds = [] ; As = [] ; Rs = [] ; Ss = []
    Ps = [] ; Ns = []
    for k in range(k_fold) :
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        model = load_model(LR, DRO, I)
        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        y_te = Y_vraie[te_indice]
        y_tr = Y_vraie[tr_indice]
        x_te = X_vraie[te_indice]
        x_tr = X_vraie[tr_indice]
         
        model.fit(x_tr, y_tr, batch_size=5, epochs=EPOCH)
        params = len(model.trainable_weights)
        model.trainable = False
                
        D, A, R, S, P, N = results(model, x_te, y_te, TR) 

        Ds.append(D); As.append(A);Rs.append(R);Ss.append(S);
        Ps.append(P); Ns.append(N);
    D_m = sum(Ds)/len(Ds) ; A_m = sum(As)/len(As) ; R_m = sum(Rs)/len(Rs)
    S_m = sum(Ss)/len(Ss) ; P_m = sum(Ps)/len(Ps) ; N_m = sum(Ns)/len(Ns)
   
    # Compute the mean and std of all the metrics defined above.
    npD = np.array(Ds) ; npA = np.array(As) ; npR = np.array(Rs) ; npS = np.array(Ss) ; 
    npP = np.array(Ps) ; npN = np.array(Ns)

    D_m = np.mean(npD) ; A_m = np.mean(npA) ; R_m = np.mean(npR)
    S_m = np.mean(npS) ; P_m = np.mean(npP) ; N_m = np.mean(npN)

    D_m2 = np.std(npD) ; A_m2 = np.std(npA) ; R_m2 = np.std(npR)
    S_m2 = np.std(npS) ; P_m2 = np.std(npP) ; N_m2 = np.std(npN)
 	
    values = [I, EPOCH, LR, DRO, TR, D_m, D_m2, A_m, A_m2, R_m,R_m2, S_m,S_m2, P_m,P_m2, N_m,N_m2, params]
   
    
    return values
    
# Load the 48 real images of satellite streaks after Data Augmentation.
X_vraie = np.load('FineTuningData/X_augmented.npy')
Y_vraie = np.load('FineTuningData/Y_augmented.npy')


all_results = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#The values of I indicated which layers to freeze during the FineTuning
I = [0,1,2]

EPOCH = [40,60,80]

TR = [0.15, 0.25, 0.3]

DRO = [0.1, 0.2]

LR = [1e-2, 5*1e-3, 1e-4]

for mod in I : 
    for ep in EPOCH : 
        for tresh in TR : 
            for drop in DRO : 
                for lr in LR : 
                    # Train the model and save the results for each combination
                    results = main(mod, ep, lr, tresh, drop, X_vraie, Y_vraie)
		    

                    all_results = np.vstack((all_results, results))
np.save('FineTuningData/Fine_Tuning_Parameters.npy',all_results)

    