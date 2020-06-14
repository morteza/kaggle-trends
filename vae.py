#%%

import streamlit as st
import pathlib
import datetime

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from IPython.display import display

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, Embedding, Dropout
#from tensorflow.keras.layers import Conv1DTranpose

from tensorflow.keras.callbacks import TensorBoard


import matplotlib.pyplot as plt

# Network parameter
n_embeddings = 10
data_dir = pathlib.Path('./data')


# load dataset
fnc_df = pd.read_csv(data_dir / 'fnc.csv')
n_features = len(fnc_df.columns)

#((trainX, _), (testX, _)) = mnist.load_data()
X_train, X_test = train_test_split(fnc_df)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

dataset = tf.data.Dataset.from_tensor_slices(X_train.values)

model = Sequential([
  #Flatten(input_shape=(n_features,)),
  Conv1D(128, 3,activation='relu'),
  Dense(units=n_embeddings, activation='relu'),
  Dropout(0.2),
  Dense(units=n_features, activation='sigmoid')
])

#model.build()
#model.summary()


model.compile(optimizer='adam', loss='mse')


log_dir = "/tmp/logs/trends_vae/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

history = model.fit(x=X_train, 
          y=X_train, 
          epochs=10,
          validation_data=(X_test, X_test), 
          callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=1)])


embeddings = model.weights[0].numpy()


N = np.arange(0, 10)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
