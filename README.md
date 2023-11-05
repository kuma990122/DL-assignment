# DL-assignment
Image classification using pretrained convolutional networks

Team name: KuMua

Team member:  Xue Zhexiong

Neptun code: CHQF4Y

Project description:  For this topic, I'm going to use a pre-trained convolutional network(GoogleNet) to implement the image classification. According to the requirement of the topic, there is a comparison between the GoogleNet and other randomly initialized neural network going to be shown. The comparison is about the time complexity of training, the accuracy of image classification. The dataset for this project would be CIFAR-10, which is a small dataset, sum up all the samples, there is only 60000 images for training and testing.

functions of the files:

dataset(folder): Includes the dataset, test set for training our model and testing

main.py(python file): Includes the code for implementing the application for image classification and the comparison between our model and other CNN models

## Software environment
Using Pycharm with Jupyter plugin for developing the baseline model, the file itself can be loaded into colab and run with T4 GPU, but running the trainning of the model may cost lots of time.

## Pipeline
import packages → download dataset → transform the dataset into appropriate size for fitting model → load the dataset into dataloader → load the pretrained model → set the appropriate optimizer and loss function for each model → load the data into GPU, clear the gradient data, train the model with data, record the loss value and doing backward propagation, updating with optimizer which we set before → output the loss value, accuracy, F1-score, confusion matrix

## What to run for the training
.IPYNB with results

## what to run for evaluation  
.IPYNB with images

