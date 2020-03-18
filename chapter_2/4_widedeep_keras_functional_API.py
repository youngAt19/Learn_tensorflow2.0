# -*- coding: utf-8 -*-

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow import keras

#%%
from sklearn.datasets import load_boston
housing = load_boston()

#%%
from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
        housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_all, y_train_all, random_state=11)

#%%
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#%%
# function style API
# f(x) = h(g(x))
input = keras.layers.Input(shape=x_train.shape[1:])
#deep model
hidden1 = keras.layers.Dense(20, activation='relu')(input)
hidden2 = keras.layers.Dense(10, activation='relu')(hidden1)

#concat with wide model
concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(
        patience=5,min_delta=1e-3)]

#%%
history = model.fit(x_train_scaled, y_train,
                    validation_data=(x_valid_scaled, y_valid),
                    epochs=100,
                    callbacks=callbacks)

#%%
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

#%%
model.evaluate(x_test_scaled, y_test)