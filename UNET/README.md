##Fine_Tuning.ipynb
This jupyter notebook loads the models after and before Fine-Tuning and compares the results.

## DATA
The folder FineTuningData contains the true images of satellite streaks, before and after data augmentation. 
The folder CNN_trainingSet is empty as it is too heavy, but should contain the data to train the U-NET from scratch.
The checkpoints of model1 are the checkpoints of the pre-trained model on synthetic images, and the checkpoints of model2 are 
the results after Fine-Tuning on the best parameters.

## Training the U-NET
`python CNN_train.py`
It calls without arguments and trains the UNET. 


## Choosing the best parameters for Fine-Tuning
`python FineTuning.py`
It calls without arguments. It trains 163 differents models, starting from the checkpoints from the pre-trained U-NET.
It saves the result of each model in 'Fine_Tuning_Parameters.npy'