import tensorflow as tf

class SiameseNN:

    # Create Model
    def __init__(self, num_of_input_features):
        self.input1 = tf.placeholder(tf.float32, [None, num_of_input_features])
        self.input2 = tf.placeholder(tf.float32, [None, num_of_input_features])

        with tf.variable_scope("SiameseNN") as scope:
            self.output1 = self.create_network(self.input1)
            scope.reuse_variables()
            self.output2 = self.create_network(self.input2)

        # Define loss