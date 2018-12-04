import tensorflow as tf
import numpy as np
import os
from helper_function import parse_json
import SiameseNN as snn

saver = tf.train.Saver()

init = tf.global_variables_initializer()

data_X, data_y = parse_json('../../datasets/tapping_behaviour.json')  # to parse the json data

people = snn.SiameseNN(18)
with tf.Session() as sess:
    sess(init)
    saver.restore(sess, 'saved_models/model.ckpt')
    embed = people.output1.eval({people.input1: data_X[0]})
    embed2 = people.output2.eval({people.input2: data_X[1]})

    print('data 1 = ', embed)
    print('data 2 = ', embed2)

