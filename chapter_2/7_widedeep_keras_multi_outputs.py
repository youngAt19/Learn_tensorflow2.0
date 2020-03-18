# -*- coding: utf-8 -*-

#%%
import matplotlib.pyplot as plt
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
#multi input
input_wide = keras.layers.Input(shape=[7])
input_deep = keras.layers.Input(shape=[8])
hidden1 = keras.layers.Dense(20, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(20, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs=[input_wide, input_deep],
                          outputs=[output, output2])

model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(
        patience=5,min_delta=1e-3)]

#%%
x_train_scaled_wide = x_train_scaled[:, :7]
x_train_scaled_deep = x_train_scaled[:, 5:]
x_valid_scaled_wide = x_valid_scaled[:, :7]
x_valid_scaled_deep = x_valid_scaled[:, 5:]
x_test_scaled_wide = x_test_scaled[:, :7]
x_test_scaled_deep = x_test_scaled[:, 5:]


history = model.fit([x_train_scaled_wide, x_train_scaled_deep], 
                    [y_train, y_train],
                    validation_data=([x_valid_scaled_wide,
                                      x_valid_scaled_deep],
                                     [y_valid, y_valid]),
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
model.evaluate([x_test_scaled_wide, x_test_scaled_deep], [y_test, y_test])