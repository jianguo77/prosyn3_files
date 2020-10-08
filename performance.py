# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:48:07 2020

@author: Nisha Haulkhory
"""

from keras.models import model_from_json
import numpy as np
import pandas as pd

model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights("fer.h5")

df_PrivateTest = pd.read_csv('df_PrivateTest.csv')
df_PublicTest = pd.read_csv('df_PublicTest.csv')
df_training = pd.read_csv('df_training.csv')


def fer2013(dataset):
    """Transforms the (blank separated) pixel strings in the DataFrame to an 3-dimensional array 
    (1st dim: instances, 2nd and 3rd dims represent 2D image)."""
    
    data = []
    pixels_list = dataset["pixels"].values
    
    for pixels in pixels_list:
        single_image = np.reshape(pixels.split(" "), (48, 48)).astype("float")
        data.append(single_image)
        
    # Convert list to 4D array:
    data = np.expand_dims(np.array(data), -1)
    
    # Normalize image data:
   # X -= np.mean(X, axis=0)
    data = data/255
    
    return data

X_PrivateTest = fer2013(df_PrivateTest)
X_train = fer2013(df_training)
X_PublicTest = fer2013(df_PublicTest)

Y_train = pd.get_dummies(df_training['emotion']).values
Y_PrivateTest = pd.get_dummies(df_PrivateTest['emotion']).values
Y_PublicTest = pd.get_dummies(df_PublicTest['emotion']).values
#Y.shape
print(Y_train.shape)
print(Y_PrivateTest.shape)
print(Y_PublicTest.shape)

print(X_PrivateTest.shape)
print(X_train.shape)
print(X_PublicTest.shape)

#Compile the model
optimizer = 'adam'
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
results = model.evaluate(X_PrivateTest, Y_PrivateTest)
print("test loss, test acc:", results)
