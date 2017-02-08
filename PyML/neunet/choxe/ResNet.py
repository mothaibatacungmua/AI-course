import numpy as np
import tensorflow as tf
from choxe import load_data
from CNN_BN import AvgPooling2D, MaxPooling2D, SoftmaxLayer, batch_norm, \
            FullConnectedLayer, evaluation, Convl2D

from math import floor
'''
ref: https://github.com/tensorflow/models/blob/master/resnet
'''

'''
Related Papers:
[1] https://arxiv.org/pdf/1603.05027v2.pdf
[2] https://arxiv.org/pdf/1512.03385v1.pdf
[3] https://arxiv.org/pdf/1605.07146v1.pdf
'''

RELU_LEAKINESS = 0.1

#http://cs231n.github.io/neural-networks-1/
def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x,0.0), leakiness * x, x, name='leaky_relu')

# See Normal Residual block and Bottleneck residual block in Figure 5 in the paper [2]
def residual(x, nin_feature_maps, nout_feature_maps,
             strides, activate_before_residual=False,
             phase_train=True):
    '''Residual unit with 2 sub layers'''
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            x = batch_norm(x, phase_train)
            x = leaky_relu(x, RELU_LEAKINESS)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = batch_norm(x, phase_train)
            x = leaky_relu(x, RELU_LEAKINESS)

    with tf.variable_scope('sub1'):
        cv = Convl2D(x, nin_feature_maps, nout_feature_maps, (3,3), strides=strides)
        x = cv.output()

    with tf.variable_scope('sub2'):
        x = batch_norm(x, phase_train)
        x = leaky_relu(x, RELU_LEAKINESS)
        cv = Convl2D(x, nout_feature_maps, nout_feature_maps, (3,3), strides=(1,1))
        x = cv.output()

    with tf.variable_scope('sub_add'):
        if nin_feature_maps != nout_feature_maps:
            sub_pool = AvgPooling2D(orig_x, [1, strides[0], strides[1], 1], padding="VALID")
            orig_x = sub_pool.output()
            #Note that orig_x is a 4D tensor, last index is number of feature maps
            orig_x = tf.pad(
                orig_x, [[0,0],[0,0],[0,0],
                         [(nout_feature_maps - nin_feature_maps)//2, (nout_feature_maps - nin_feature_maps)//2]]
            )

    x += orig_x
    return x

    pass

def bottleneck_residual(x, nin_feature_maps, nout_feature_maps,
                        strides, activate_before_residual=False,
                        phase_train=True):
    '''Bottleneck residual unit with 3 sub layers'''
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            x = batch_norm(x, phase_train)
            x = leaky_relu(x, RELU_LEAKINESS)
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            orig_x = x
            x = batch_norm(x, phase_train)
            x = leaky_relu(x, RELU_LEAKINESS)

    with tf.variable_scope('sub1'):
        cv = Convl2D(x, nin_feature_maps, nout_feature_maps/4, (1,1), strides=strides)
        x = cv.output()

    with tf.variable_scope('sub2'):
        x = batch_norm(x, phase_train)
        x = leaky_relu(x, RELU_LEAKINESS)
        cv = Convl2D(x, nout_feature_maps/4, nout_feature_maps/4, (3,3), strides=(1,1))
        x = cv.output()

    with tf.variable_scope('sub3'):
        x = batch_norm(x, phase_train)
        x = leaky_relu(x, RELU_LEAKINESS)
        cv = Convl2D(x, nout_feature_maps/4, nout_feature_maps, (1,1), strides=(1,1))
        x = cv.output()

    with tf.variable_scope('sub_add'):
        if nin_feature_maps != nout_feature_maps:
            cv = Convl2D(orig_x, nin_feature_maps, nout_feature_maps, (1,1),strides=strides)
            orig_x = cv.output()

    x += orig_x

    return x
    pass

USING_BOTTLENECK = True
NUM_RESIDUAL_UNITS = 2

def ResNet(X, y, dropout_prob, phase_train):
    x = tf.reshape(X, [-1, 256, 256, 3])

    activate_before_residual = [True, False]

    if USING_BOTTLENECK:
        res_func = bottleneck_residual
        filters = [64, 128, 256]
    else:
        res_func = residual
        filters = [64, 64, 128]

    with tf.variable_scope('conv_1'):
        conv1 = Convl2D(x, 3, 64, (7, 7), strides=(2, 2), padding='SAME')
        conv1_bn = batch_norm(conv1.output(), phase_train)
        conv1_out = leaky_relu(conv1_bn, RELU_LEAKINESS)

        pool1 = MaxPooling2D(conv1_out, [1, 2, 2, 1], padding='SAME')
        x = pool1.output()

    with tf.variable_scope('unit1_0'):
        x = res_func(x, filters[0], filters[1], (1,1), activate_before_residual[0])

    for i in range(1, NUM_RESIDUAL_UNITS):
        with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1],(2,2), False)

    with tf.variable_scope('unit2_0'):
        x = res_func(x, filters[0], filters[1], (1, 1), activate_before_residual[1])

    for i in range(1, NUM_RESIDUAL_UNITS):
        with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], (2, 2), False)

    with tf.variable_scope('last_unit'):
        x = batch_norm(x, phase_train)
        x = leaky_relu(x, RELU_LEAKINESS)

        output_size = x.get_shape()[1:2]
        x = tf.reshape(x, [-1, output_size[0] * output_size[1] * filters[1]])

    #Hidden Layer: full connected layer with dropout
    with tf.variable_scope('fc1'):
        fc = FullConnectedLayer(x, output_size[0] * output_size[1] * filters[1], 1000)
        x = fc.output()
        fc_dropped = tf.nn.dropout(x, dropout_prob)

    #Final Layer: Softmax classification
    y_pred = SoftmaxLayer(fc_dropped, 1000, 3).output()
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

    loss = cross_entropy
    accuracy = evaluation(y_pred, y)

    return loss, accuracy, y_pred
    pass

SAVE_SESS = 'resnet_choxe.ckpt'
BATCH_SIZE = 100

if  __name__ == '__main__':
    TASK = 'train'
    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    dropout_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, accuracy, y_pred = ResNet(X, y, dropout_prob, phase_train)

    #Train model
    learning_rate = 0.01
    train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    #vars_to_save = tf.trainable_variables()

    #Need to save the moving average variables


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

                train_op.run({X: batch_x, y: batch_y, dropout_prob: 0.5, phase_train: True})

                if i % 200 == 0:
                    cv_fd = {X: batch_x, y: batch_y, dropout_prob: 1.0, phase_train: False}
                    train_loss = loss.eval(cv_fd)
                    train_accuracy = accuracy.eval(cv_fd)

                    print 'Step, loss, accurary = %6d: %8.4f, %8.4f' % (i,train_loss, train_accuracy)

        '''
        print '\nTesting...'
        test_fd = {X: test_X, y: test_y, dropout_prob: 1.0, phase_train: False}
        print(' accuracy = %8.4f' % accuracy.eval(test_fd))
        '''

        # Save the session
        if TASK == 'train':
            saver.save(sess, SAVE_SESS)