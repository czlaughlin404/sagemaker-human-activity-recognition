#!/usr/bin/env python

import argparse
import os
import numpy as np
from numpy import load

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras import backend as K


# fit and evaluate a model
def execute_model(trainX, trainy, testX, testy):
    
    n_features, n_outputs = trainX.shape[1], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4,32
    
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    
    # define model
    
    #https://keras.io/api/models/sequential/
    model = Sequential()
    
    #https://keras.io/api/layers/recurrent_layers/time_distributed/
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    #https://keras.io/api/layers/recurrent_layers/lstm/
    model.add(LSTM(128))
    model.add(Dropout(rate=0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.summary()
      
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    #https://keras.io/api/losses/
    #https://keras.io/api/optimizers/
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=2)
  
    # evaluate model
    loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
    
     # save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
    
    return loss, accuracy

# run an experiment
def run_experiment():
    
    trainX = np.load(os.path.join(training_dir, 'train_data.npy'))
    trainy = np.load(os.path.join(training_dir, 'train_labels.npy'))
    testX = np.load(os.path.join(test_dir, 'test_data.npy'))
    testy = np.load(os.path.join(test_dir, 'test_labels.npy'))
              
    loss, accuracy = execute_model(trainX, trainy, testX, testy)
    print('Accuracy = %.3f' % (accuracy))
    print('Loss = %.3f' % (loss))
    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu_count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args, _ = parser.parse_known_args()
    
    epochs = args.epochs
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    test_dir = args.test
    
    run_experiment()
    