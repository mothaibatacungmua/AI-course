import numpy as np
import tensorflow as tf
from CNN_BN import AvgPooling2D, MaxPooling2D, SoftmaxLayer, batch_norm, \
            FullConnectedLayer, evaluation, Convl2D
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

def ResNet():
    pass