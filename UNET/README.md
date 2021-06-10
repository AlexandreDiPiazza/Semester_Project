# Performance of the model
We train here the U-NET on the synthetic images, and fine-tune our model on images containing real slow satellite streaks.

## Illustrating the results 
`Fine_Tuning.ipynb`
This jupyter notebook loads the 2 models after and before Fine-Tuning and compares the results on the prediction of 
satellite streaks on real images.

## DATA
The folder FineTuningData contains the true images of satellite streaks, before and after data augmentation. 
The folder CNN_trainingSet is empty as it is too heavy, but should contain the data to train the U-NET from scratch. This data
can be directly created by running the corresponding python file in the other folder (GAN).
The checkpoints of model 1 are the checkpoints of the pre-trained model on synthetic images, and the checkpoints of model 2 are 
the results after Fine-Tuning on the best parameters. It allows to directly load the parameters of each model.

## Training the U-NET
`python CNN_train.py`
It calls without arguments and trains the UNET on the synthetic data.


## Choosing the best parameters for Fine-Tuning
`python FineTuning.py`
It calls without arguments. It trains 163 differents models, starting from the checkpoints from the pre-trained U-NET (model 1).
It saves the result of each new model in 'Fine_Tuning_Parameters.npy' to measure the model that performed best.