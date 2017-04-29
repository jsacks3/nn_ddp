# Module: network.py
# Created on: 4/27/17
# Author: Jake Sacks

import numpy as np
import json
import io
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# parameters
arch = [200, 200]
epochs = 100
batch_size = 1
mse = 0
mape = 0

def mean_absolute_percentage_error(y_true, y_pred):
    err = np.zeros([y_true.shape[1], 1])
    for i in range(0, y_true.shape[1]):
        for j in range(0, y_true.shape[0]):
            if (y_true[j,i] != 0):
                err[i] += np.abs((y_true[j,i] - y_pred[j,i]) / y_true[j,i])/y_true.shape[0]
    return err

def load(filename):
    # create network architecture
    net = Sequential()
    net.add(Dense(arch[0], activation='relu', input_dim=5))
    net.add(Dropout(0.01))
    net.add(Dense(arch[1], activation='relu'))
    net.add(Dropout(0.01))
    net.add(Dense(4))

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    net.compile(optimizer=opt, loss='mse')

    # load weights
    if filename != None:
        file = io.open(filename, 'r', encoding='utf8')
        json_data = file.read()
        data = json.loads(json_data)
        W  = [data['layer_0_weights']]
        W.append(data['layer_0_bias'])
        W.append(data['layer_1_weights'])
        W.append(data['layer_1_bias'])
        W.append(data['layer_2_weights'])
        W.append(data['layer_2_bias'])
        net.set_weights(W)

    return net

def train(net, x_train, y_train):
    net.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return net

def evaluate(net, x_test, y_test):
    global mse, mape

    mse = net.evaluate(x_test, y_test, batch_size=batch_size)
    print('\nMSE = ' + str(mse))

    y = net.predict(x_test, batch_size=batch_size)
    mape = mean_absolute_percentage_error(y_test, y)
    print('MAPE = ')
    print(mape)

def save(net, filename):
    W = net.get_weights()
    data = {'hidden_layer_sizes': arch,
            'MSE': mse,
            'MAPE': mape.tolist(),
            'layer_0_weights': W[0].tolist(),
            'layer_0_bias': W[1].tolist(),
            'layer_1_weights': W[2].tolist(),
            'layer_1_bias': W[3].tolist(),
            'layer_2_weights': W[4].tolist(),
            'layer_2_bias': W[5].tolist()}

    json_data = json.dumps(data, separators=(',', ':'), sort_keys=False, indent=4)
    file = io.open(filename, 'w', encoding='utf8')
    file.write(str(json_data))
