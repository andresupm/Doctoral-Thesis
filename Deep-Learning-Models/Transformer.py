# -*- coding: utf-8 -*-
"""MULTIVARIATE_TRAFO_FORECASTING_ANDRES_MATERNAv6.ipynb"""

from google.colab import drive
drive.mount('/gdrive')

import pandas as pd
from pandas import read_csv
import glob
path =r'/gdrive/My Drive/Materna-Trace-1' # use your path
allFiles = glob.glob(path + "/*")

list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,delimiter=";",decimal=',')
    list_.append(df)

df = pd.concat(list_, axis = 0, ignore_index = True)

len(allFiles)

df.shape

dataset=df

dataset.drop('Timestamp', axis=1, inplace=True)
dataset.drop('CPU cores', axis=1, inplace=True)
dataset.drop('CPU capacity provisioned [MHZ]', axis=1, inplace=True)
dataset.drop('CPU usage [MHZ]', axis=1, inplace=True)
dataset.drop('Memory capacity provisioned [KB]', axis=1, inplace=True)
dataset.drop('Memory usage [KB]', axis=1, inplace=True)
#dataset.drop('Disk write throughput [KB/s]', axis=1, inplace=True)
dataset.drop('Disk size [GB]', axis=1, inplace=True)
dataset.drop('Network transmitted throughput [KB/s]', axis=1, inplace=True)
dataset.drop('Disk read throughput [KB/s]', axis=1, inplace=True)

print(dataset.head(5))
# save to file
dataset.to_csv('materna.csv')

dataset.head(3)

from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('materna.csv', header=0, index_col=0)
print(dataset.head(5))
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()

"""# Multivariate LSTM Forecast Model

### MULTI STEP
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('materna.csv', header=0, index_col=0)
print(dataset.dtypes)
values = dataset.values
# integer encode direction
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

#scaled

"""# specify the number of time steps (aka hours)"""

n_hours = 3 # NUMBER STEPS
n_features = 4
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

reframed.head(3)

reframed.shape

# split into train and test sets
values = reframed.values

n_train_hours = 3320945


train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print(train.shape)
print(test.shape)

#train_X[1]

#train_y[1]

train[0]

# split into input and outputs
n_obs = n_hours * n_features
print(n_obs)
train_X, train_y = train[:, :n_obs], train[:, n_obs:]
test_X, test_y = test[:, :n_obs], test[:,n_obs:]
#test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

from tensorflow.keras import optimizers



from tensorflow import keras
from tensorflow.keras import layers

"""We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`.
"""

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(4, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

input_shape = train_X.shape[1:]
print(input_shape)

input_shape = train_X.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=2,
    num_transformer_blocks=4,
    mlp_units=[128,128,32],
    mlp_dropout=0.2,
    dropout=0.1,
)

learning_rate=0.0013426#0.00020972#1e-3#0.00005#0.0011145180231844202#0.00005
optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=1.)
model.compile(loss='mae', optimizer = optimizer, metrics=['mse'])
model.summary()

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto',restore_best_weights=True) 

# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=1024, validation_data=(test_X, test_y),callbacks=[monitor], verbose=1, shuffle=True)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

"""# PREDICT"""
"""### EJEMPLO Cargar modelo HFS5"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model

from tensorflow.keras.callbacks import TensorBoard

from tensorflow import keras
from tensorflow.keras.layers import Conv1D,Dropout,MaxPooling1D,Dense,Flatten, Activation,GlobalAveragePooling1D
from tensorflow.keras import layers
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
import time
import pickle
from tensorflow.keras import optimizers
from keras.models import load_model

"""## Select a trained model"""

model_name='my_model_TRAFO_MATERNAV6.h5'
print(model_name)

model = load_model(model_name)
model.summary()

"""## Reload data"""

# split into input and outputs
n_obs = n_hours * n_features
print(n_obs)
train_X, train_y = train[:, :n_obs], train[:, n_obs:]
test_X, test_y = test[:, :n_obs], test[:,n_obs:]
#test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

score = model.evaluate(test_X, test_y, verbose = 0) 

print('Test loss:', score[0]) 
print('Test MSE:', score[1])

## PREDICTED FEATURE
predictedFeat=0 #del 0 al 3

# make a prediction
yhat = model.predict(test_X)
print(test_X.shape)

from sklearn.metrics import mean_squared_error
print('MSE: ', mean_squared_error(test_y, yhat))
print('RMSE: ', mean_squared_error(test_y, yhat,squared=False))

#Reshape
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
print(test_X.shape)
print(test_X[:,-3:].shape)
print(yhat[:,[3]].shape)

# invert scaling for forecast
inv_yhat = concatenate((yhat[:,[predictedFeat]], test_X[:, -3:]), axis=1) # -(numero de features -1)----ACA EL FEATURE A PREDECIR <---> yhat[:,[3]]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y[:,[predictedFeat]]#.reshape((len(test_y), 1))#----ACA EL FEATURE A PREDECIR <---> yhat[:,[3]]
inv_y = concatenate((test_y, test_X[:, -3:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,5)

plt.plot(inv_y[0:50000],color='blue')
plt.plot(inv_yhat[0:50000],color='red')

# calculate MSE
mse = (mean_squared_error(inv_y, inv_yhat))
print('Test MSE: %.3f' % mse)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

