# Created by Ridwan Afwan Karim Fauzi
# https://www.github.com/ridwanakf

### Libraries and Modules ###
import tensorflow as tf
import numpy as np
#from sklearn.model_selection import train_test_split
import time

import SiameseNN as snn
from helper_function import parse_json
from helper_function import get_batch
from helper_function import get_label

### Hyperparameters ###
iteration = 2000000
learning_rate = 0.001
num_of_input_features = 54


### Directories ###
dir_dataset = '../../datasets/speedhack.json'
dir_model = 'saved_models/flex/model_new.ckpt'

data_X, data_y = parse_json(dir_dataset)  # to parse the json data

# Uncomment if you want to split the data.
# For the time being, due to minimum data, i do not split the dataset
# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1)

### Setup The Model ###
people = snn.SiameseNN(num_of_input_features)  # Model instance
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(people.loss)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
	
### Run TensorFlow Session ###
with tf.Session(config=config) as sess:
    sess.run(init)

    ### Load Previous Checkpoint or Not ###
    load = False # change this to True if you want to resume training
    if load:
        saver.restore(sess, dir_model)
        print("LOADED")
    else:
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        date = year + '_' + month + '_' + day + '_' + hour + '_' + minute
        dir_model = 'saved_models/model_' + date + '_.ckpt'

    # Start the training
    for step in range(iteration):

        batch_x1, batch_y1 = get_batch(10, data_X, data_y)
        batch_x2, batch_y2 = get_batch(10, data_X, data_y)
        batch_y = get_label(batch_y1, batch_y2)
        
        # percentage for similar vs not similar pair
        percentage = sum(batch_y) / len(batch_y)

        _, loss = sess.run([train, people.loss], feed_dict={
                           people.input1: batch_x1, people.input2: batch_x2, people.y_true: batch_y})

        if np.isnan(loss):
            print('Model is not converged (loss = NaN)')
            break

        if step % 10 == 0:
            print('step %d: loss: %.3f | A|P: %.3f | A|N: %.3f' %
                  (step, loss, (percentage*100), ((1-percentage)*100)))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, dir_model)
            print('Saved')

