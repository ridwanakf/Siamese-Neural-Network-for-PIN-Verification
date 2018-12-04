import tensorflow as tf

class SiameseNN:

    # Create Model
    def __init__(self, num_of_input_features):
        self.input1 = tf.placeholder(tf.float32, [None, num_of_input_features], name='input1')
        self.input2 = tf.placeholder(tf.float32, [None, num_of_input_features], name='input2')

        with tf.variable_scope('SiameseNN') as scope:
            self.output1 = self.create_network(self.input1)
            scope.reuse_variables()
            self.output2 = self.create_network(self.input2)

        # Define loss
        self.y_true = tf.placeholder(tf.float32, [None])
        self.loss = self.compute_loss(bias = 5.0)

    # Setup Networks
    def create_network(self, input):
        dense1 = self.dense(input, 1024, 'dense1')
        activation_func1 = tf.nn.relu(dense1)

        dense2 = self.dense(activation_func1, 1024, 'dense2')
        activation_func2 = tf.nn.relu(dense2)

        dense3 = self.dense(activation_func2, 2, 'dense3')
        
        return dense3

    # Setup Dense Neurons
    def dense(self, input_layer, num_of_weights, name):
        # Check the dimension of input tensor
        assert len(input_layer.get_shape()) == 2
        num_of_prev_weight = input_layer.get_shape()[1]
        init_normal = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'_W', dtype=tf.float32,
                            shape=[num_of_prev_weight, num_of_weights], initializer=init_normal)
        b = tf.get_variable(name+'_b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[num_of_weights], dtype=tf.float32))
        xW = tf.matmul(input_layer, W)
        dense = tf.nn.bias_add(xW, b)
        return dense

    # Compute the loss function
    def compute_loss(self, bias):
        # using contrastive loss function

        margin = tf.constant(bias, name='margin')
        # label for positive (similar) inputs
        pos_label = self.y_true
        # label for negatif (not similar) inputs
        neg_label = tf.subtract(1.0, self.y_true, name='1-y')

        feature_distance = tf.pow(tf.subtract(self.output1, self.output2), 2)
        feature_distance = tf.reduce_sum(feature_distance, 1, name='dist2')
        sqrt_feature_distance = tf.sqrt(feature_distance+1e-6, name='dist')

        positive = tf.multiply(pos_label, feature_distance, name='pos')
        negative = tf.multiply(neg_label, tf.pow(tf.maximum(tf.subtract(margin, sqrt_feature_distance), 0.0), 2), name='neg')
        losses = tf.add(positive,negative, name='losses')
        loss = tf.reduce_mean(losses, name='loss')
        
        return loss
