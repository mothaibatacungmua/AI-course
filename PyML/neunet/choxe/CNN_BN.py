import os
import numpy as np
import tensorflow as tf
from choxe import load_data
from math import ceil, floor

class Convl2D(object):
    def __init__(self,
                 input,
                 nin_feature_maps,
                 nout_feature_maps,
                 filter_shape,
                 strides=(1,1),
                 activation='relu',
                 using_bias=False,
                 padding='SAME'):
        self.input = input
        self.nin_feature_maps = nin_feature_maps
        self.activation = activation
        self.using_bias = using_bias
        self.strides = strides
        self.padding = padding

        wshape = [filter_shape[0], filter_shape[1], nin_feature_maps, nout_feature_maps]

        weights = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), trainable=True)
        if self.using_bias:
            bias = tf.Variable(tf.constant(0.1, shape=[nout_feature_maps]), trainable=True)
            self.bias = bias

        self.weights = weights
        if self.using_bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]
    pass

    def output(self):
        linout = tf.nn.conv2d(self.input, self.weights,
                              strides=[1, self.strides[0], self.strides[1], 1],
                              padding=self.padding)
        if self.using_bias:
            linout = linout + self.bias

        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        else:
            self.output = linout

        return self.output

class MaxPooling2D(object):
    def __init__(self, input, pooling_size, padding='SAME'):
        self.input = input
        if pooling_size == None:
            pooling_size = [1, 2, 2, 1]

        self.pooling_size = pooling_size
        self.padding = padding

    def output(self):
        # pooling_size = stride_size
        strides = self.pooling_size
        self.output = tf.nn.max_pool(self.input, ksize=self.pooling_size,
                                     strides=strides, padding=self.padding)

        return self.output
    pass

class AvgPooling2D(object):
    def __init__(self, input, pooling_size, padding='SAME'):
        self.input = input
        if pooling_size == None:
            pooling_size = [1, 2, 2, 1]

        self.pooling_size = pooling_size
        self.padding = padding

    def output(self):
        strides = self.pooling_size
        self.output = tf.nn.avg_pool(self.input, ksize=self.pooling_size,
                                     strides=strides, padding=self.padding)

        return self.output
    pass

class FullConnectedLayer(object):
    def __init__(self, input, n_in, n_out, activation='relu'):
        self.input = input
        self.activation = activation
        weights = tf.Variable(tf.truncated_normal([n_in, n_out], mean=0.0,
                                                  stddev=0.05), trainable=True)
        bias = tf.Variable(tf.zeros([n_out]), trainable=True)

        self.weights = weights
        self.bias = bias

        self.params = [self.weights, self.bias]

    def output(self):
        z = tf.matmul(self.input, self.weights) + self.bias
        if self.activation == 'relu':
            self.output = tf.nn.relu(z)
        elif self.activation == 'tanh':
            self.output = tf.nn.tanh(z)
        elif self.activation == 'sigmoid':
            self.output = tf.nn.sigmoid(z)

        return self.output

    pass

class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        weights = tf.Variable(tf.random_normal([n_in, n_out],mean=0.0, stddev=0.05), trainable=True)
        bias = tf.Variable(tf.zeros([n_out]), trainable=True)

        self.weights = weights
        self.bias = bias

        self.params = [self.weights, self.bias]

    def output(self):
        z = tf.matmul(self.input, self.weights) + self.bias
        self.output = tf.nn.softmax(z)

        return self.output
    pass

'''
paper: https://arxiv.org/abs/1502.03167
ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
'''
def batch_norm(X, phase_train):
    with tf.variable_scope('bn'):
        params_shape = [X.get_shape()[-1]]
        beta = tf.Variable(tf.constant(0.0, shape=params_shape),
                           name='beta', trainable=True)

        gamma = tf.Variable(tf.constant(0.0, shape=params_shape),
                            name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(X, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5) # see section 3.1 in the paper

        def moving_avg_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            moving_avg_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)

    return normed

def evaluation(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return accuracy

def cnn_bn_model(X, y, dropout_prob, phase_train):
    '''
    + -1 for size of mini-batch,
    + 256x256 is size of image
    + 3 is three channels RBG
    '''
    X_image = tf.reshape(X, [-1, 256, 256, 3])

    '''
    output height and width padding "SAME":
        out_height = ceil(float(in_height) / float(strides[1]))
        out_width  = ceil(float(in_width) / float(strides[2]))

    output height and width padding "VALID":
        out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

    ref: https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
    '''
    def compute_output_size(input_size, filter_size, strides, padding):
        if padding == 'SAME':
            out_height = ceil(float(input_size[0]) / float(strides[0]))
            out_width = ceil(float(input_size[1]) / float(strides[1]))
        else: #padding == 'VALID'
            out_height = ceil(float(input_size[0] - filter_size[0] + 1) / float(strides[0]))
            out_width = ceil(float(input_size[1] - filter_size[1] + 1) / float(strides[1]))

        return (int(out_height), int(out_width))


    # Layer 1, convol window 7x7, stride width = 2, stride height = 2 with max pooling, padding = SAME
    # Number input channels = 3, number output channels = 64
    with tf.variable_scope('conv_1'):
        conv1 = Convl2D(X_image, 3, 64, (7, 7), strides=(2,2), padding='SAME')
        conv1_bn = batch_norm(conv1.output(), phase_train)
        conv1_out = tf.nn.relu(conv1_bn)

        pool1 = MaxPooling2D(conv1_out, [1, 2, 2, 1], padding='SAME')
        pool1_out = pool1.output()

    # Layer 2, convol window 3x3, stride width = 1, stride height = 1 without pooling, padding = SAME
    # Number output channels: 64, number output channels = 64
    with tf.variable_scope('conv_2'):
        conv2 = Convl2D(pool1_out, 64, 64, (3,3), strides=(1,1), padding='SAME')
        conv2_bn = batch_norm(conv2.output(), phase_train)
        conv2_out = tf.nn.relu(conv2_bn)

    # Layer 3, convol window 3x3, stride width = 1, stride height = 1 without pooling, padding = SAME
    # Number output channels: 64, number output channels = 64
    with tf.variable_scope('conv_3'):
        conv3 = Convl2D(conv2_out, 64, 64, (3,3), strides=(1,1), padding='SAME')
        conv3_bn = batch_norm(conv3.output(), phase_train)
        conv3_out = tf.nn.relu(conv3_bn)

    # Layer 4, convol window 3x3, stride width = 1, stride height = 1 with avg pooling, padding = VALID
    # Number output channels: 64, number output channels = 128
    with tf.variable_scope('conv_4'):
        conv4 = Convl2D(conv3_out, 64, 128, (3, 3), strides=(1, 1), padding='VALID')
        conv4_bn = batch_norm(conv4.output(), phase_train)
        conv4_out = tf.nn.relu(conv4_bn)

        pool4 = AvgPooling2D(conv4_out, [1, 2, 2, 1], padding='SAME')
        pool4_out = pool4.output()

        #print pool4_out.get_shape()
        output_size = pool4_out.get_shape()[1:3]
        flat_size = int(output_size[0])*int(output_size[1])*128
        pool4_flatten = tf.reshape(pool4_out, [-1, flat_size])

    # Layer 5: full connected layer with dropout
    with tf.variable_scope('fc1'):
        fc1 = FullConnectedLayer(pool4_flatten, flat_size, 1000)
        fc1_out = fc1.output()
        fc1_dropped = tf.nn.dropout(fc1_out, dropout_prob)

    y_pred = SoftmaxLayer(fc1_dropped, 1000, 3).output()
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

    loss = cross_entropy
    accuracy = evaluation(y_pred, y)

    return loss, accuracy, y_pred

SAVE_SESS = 'cnn_bn_choxe.ckpt'
BATCH_SIZE = 100

if  __name__ == '__main__':
    TASK = 'train'
    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    dropout_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, accuracy, y_pred = cnn_bn_model(X, y, dropout_prob, phase_train)

    #Train model
    learning_rate = 0.01
    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    #vars_to_save = tf.trainable_variables()


    init_op = tf.global_variables_initializer()
    if TASK == 'test':
        restore_sess = True
    elif TASK == 'train':
        restore_sess = False
    else:
        assert 1==0, "Task isn't supported"

    saver = tf.train.Saver()

    train_X, train_y, test_X, test_y = load_data()
    N = train_y.shape[0]
    num_batches = floor(N / BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(init_op)

        if restore_sess:
            saver.restore(sess, SAVE_SESS)

        if TASK == 'train':
            print '\nTraining...'
            for i in range(1, 10000):
                mini_batch_index = i
                if mini_batch_index > num_batches:
                    mini_batch_index = 1

                batch_x = train_X[(mini_batch_index-1)*BATCH_SIZE:mini_batch_index*BATCH_SIZE]
                batch_y = train_y[(mini_batch_index-1)*BATCH_SIZE:mini_batch_index*BATCH_SIZE]

                train_op.run(feed_dict={X: batch_x, y: batch_y, dropout_prob: 0.5, phase_train: True})

                if i % 200 == 0:
                    cv_fd = {X: batch_x, y: batch_y, dropout_prob: 1.0, phase_train: False}
                    train_loss = loss.eval(feed_dict = cv_fd)
                    train_accuracy = accuracy.eval(feed_dict = cv_fd)

                    print 'Step, loss, accurary = %6d: %8.4f, %8.4f' % (i,train_loss, train_accuracy)

        '''
        print '\nTesting...'
        test_fd = {X: test_X, y: test_y, dropout_prob: 1.0, phase_train: False}
        print(' accuracy = %8.4f' % accuracy.eval(test_fd))
        '''

        # Save the session
        if TASK == 'train':
            saver.save(sess, SAVE_SESS)
