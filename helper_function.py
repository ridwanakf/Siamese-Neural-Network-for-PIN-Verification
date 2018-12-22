# Created by Ridwan Afwan Karim Fauzi
# https://www.github.com/ridwanakf

import json
import os
import random


def parse_json(dataset_dir):
    with open(dataset_dir, 'r') as f_in:
        data = json.load(f_in)
    data = data['sensor_data']
    X = []
    y = []
    person_key = []

    num_sample = 5  # 5 sample per person
    for person in data:
        person_key.append(person)
        for i in range(num_sample):
            y.append(person)

    for person in person_key:
        json_explorer = data
        json_explorer = json_explorer[person]
        for sample in json_explorer:
            temp = []
            for value in sample['acc_angular_release']:
                temp.append(value)
            for value in sample['acc_angular_touch']:
                temp.append(value)
            for value in sample['acc_linear_release']:
                temp.append(value)
            for value in sample['acc_linear_touch']:
                temp.append(value)
            for value in sample['key_hold']:
                value = value / 1000  # ms to s
                temp.append(value)
            for value in sample['pressure_release']:
                temp.append(value)
            for value in sample['pressure_touch']:
                temp.append(value)
            for value in sample['size_release']:
                temp.append(value)
            for value in sample['size_touch']:
                temp.append(value)
            X.append(temp)

    return X, y


def get_batch(num_of_data, X, y):
    max_index = len(y) - 1
    start_index = random.randint(0, max_index - num_of_data)
    
    X_batch = []
    y_batch = []    
    for i in range(num_of_data):
        index = random.randint(0, max_index)
        X_batch.append(X[index])
        y_batch.append(y[index])
    
    return X_batch, y_batch


def get_label(batch1, batch2):
    # compare the label of 2 batches, if the same then return 1, else 0
    length = len(batch1)
    comparison = []
    
    for i in range(length):
        if batch1[i] == batch2[i]:
            comparison.append(1.0)
        else:
            comparison.append(0.0)

    return comparison
