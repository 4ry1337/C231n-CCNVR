import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def SoftmaxRegression(W, X, y, reg):
    loss = 0.0 # Initialize the loss.
    dW = np.zeros_like(W) # Initialize thegradient to zero.
    N = X.shape[0] # number of samples
    Y_hat = X @ W  # raw scores matrix
    P = np.exp(Y_hat - Y_hat.max()) # numerically stable exponents
    P /= P.sum(axis=1, keepdims=True) # row-wise probabilities (softmax)

    loss = -np.log(P[range(N), y]).sum() # sum cross entropies as loss
    loss = loss / N + reg * np.sum(W**2) # average loss and regularize 

    P[range(N), y] -= 1 # update P for gradient
    dW = X.T @ P / N + 2 * reg * W # calculate gradient
    
    return loss, dW

def SVM(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    N = len(y) # number of samples
    Y_hat = X @ W # raw scores matrix

    y_hat_true = Y_hat[range(N), y][:, np.newaxis] # scores for true labels
    margins = np.maximum(0, Y_hat - y_hat_true + 1) # margin for each score
    loss = margins.sum() / N - 1 + reg * np.sum(W**2) # regularized loss

    dW = (margins > 0).astype(int) # initial gradient with respect to Y_hat
    dW[range(N), y] -= dW.sum(axis=1) # update gradient to include correct labels
    dW = X.T @ dW / N + 2 * reg * W # gradient with respect to W

    return loss, dW

path = 'data\Agricultural-crops'
image_paths = [] #srote paths
labels = [] #store labels
classes = sorted(os.listdir(path)) #import class folders

for c in classes:
    class_folder = path + "/" + c #for each class define path
    for image in os.listdir(class_folder): 
        image_path = class_folder + "/" + image #for each image add to dataframe by path and label, i.e class
        image_paths.append(image_path)
        labels.append(c)
n_classes = len(classes) # number of classes
class_labels = np.unique(labels) #class labels
# map labels to numerical values
label_mapping = {label: i for i, label in enumerate(class_labels)}
labels = np.array([label_mapping[label] for label in labels])
df = pd.DataFrame({"image": image_paths, "label": labels}) # create dataframe
print(type(label_mapping))

