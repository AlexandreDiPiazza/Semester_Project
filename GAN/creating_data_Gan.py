import numpy as np

"""
This file load the syntehtic images and separes it into a X and Y set in order to train the CycleGAN. X contains satellite streaks, whereas
Y doesn't.
"""
datapath = 'trainset/'
name = "label"
x_train = np.load(datapath + name + '_samples.npy', allow_pickle = True)[:,0]
y_train = np.load(datapath +  name + '_targets.npy', allow_pickle = True)
hs = np.load(datapath +  name + '_patch.npy', allow_pickle = True)

indices = np.unique(np.argwhere(~np.isnan(x_train))[:,0])
x_train = x_train[indices]
y_train = y_train[indices]
hs = hs[indices]

trace = np.sum(hs)
no_trace = len(hs) - trace
        
X = np.zeros(shape = (trace, 256, 256, 3)) # 256* 256 for same size as GAN already implemented , 
X_trace = np.zeros(shape = (trace, 256, 256))

Y = np.zeros(shape = (no_trace, 256, 256, 3))  
Y_trace = np.zeros(shape = (no_trace, 256, 256))

index1 = 0
index2 = 0
for i in range(0,np.sum(hs)) : 
    if hs[i] == 1 :
        X[index1, :, :, 0] = x_train[i]
        X[index1, :, :, 1] = x_train[i]
        X[index1, :, :, 2] = x_train[i]
        X_trace[index1,:,:] = y_train[i]
        index1 += 1
    else : 
        Y[index2, :, :, 0] =  x_train[i]
        Y[index2, :, :, 1] =  x_train[i]
        Y[index2, :, :, 2] =  x_train[i]
        Y_trace[index2,:,:] = y_train[i]
        index2 += 1

np.save("X_DATA.npy", X[0:1500,:,:,:]) ; np.save("X_trace.npy", X_trace[0:1500,:,:])
np.save("Y_DATA.npy", Y[0:1500,:,:,:]) ; np.save("Y_trace.npy", Y_trace[0:1500,:,:])

                                         