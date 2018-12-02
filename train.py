### Libraries and Modules ###

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from builtins import input

import SiameseNN as snn
from helper_function import parse_json
from helper_function import get_batch

### Hyperparameters ###
iteration = 100000
learning_rate = 0.01
num_of_input_features = 52


### Directories ###
dir_dataset = ''
dir_model = 'saved_models/model.ckpt'

# TODO: implement JSON Parser method
data_X, data_y = parse_json(dir_dataset) # to parse the json data

### Setup The Model ###
people = snn.SiameseNN(num_of_input_features) # Model instance
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(people.loss)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    load = False
    if os.path.isfile(dir_model):
        key = None
        while key not in ['Y', 'y', 'n', 'N', 'Yes', 'yes', 'no', 'No']:
            key = input(
                "There is a model, do you want to load it and continue the training [Y/n]?")
        if key == 'Y' or key == 'y' or key == 'Yes' or key == 'yes':
            load = True

    # Load previous model
    if load:
        saver.restore(sess, dir_model)

    for step in range(iteration):
        # TODO: Implement batch generator on the dataset
        batch_x1, batch_y1 = get_batch(128)
        batch_x2, batch_y2 = get_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')
        
        _, loss = sess.run([train, people.loss], feed_dict={people.input1: batch_x1, people.input2: batch_x2, people.y_true: batch_y})

        if np.isnan(loss):
            print('Model is not converged (loss = NaN)')
            break

        if step % 50 == 0:
            print('step %d: loss %.3f' % (step, loss))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, dir_model)
