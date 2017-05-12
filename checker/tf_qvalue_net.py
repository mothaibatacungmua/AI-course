import sys
import numpy as np
import tensorflow as tf

class QValueNet(object):
    def __init__(self, game_size):
        self.game_size = game_size

        X = tf.placeholder(tf.float32, shape=(1, game_size * game_size), name="state")
        y = tf.placeholder(tf.float32, shape=(2), name="score")

        self.w1 = tf.get_variable("w1", 
                                shape=(self.game_size ** 2, 100), 
                                dtype=tf.float32, 
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b1 = tf.get_variable("b1", 
                                shape=(100), 
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())
        self.w2 = tf.get_variable("w2", 
                                shape=(100, 50), 
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b2 = tf.get_variable("b2",
                                shape=(50),
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())
        self.w3 = tf.get_variable("w3",
                                shape=(50, 2),
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b3 = tf.get_variable("b3",
                                shape=(2),
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

        h1 = tf.sigmoid(tf.matmul(X, self.w1) + self.b1)
        h2 = tf.sigmoid(tf.matmul(h1, self.w2) + self.b2)
        self.pred = tf.sigmoid(tf.matmul(h2, self.w3) + self.b3)
        self.cost = tf.reduce_mean(tf.squared_difference(y, self.pred))
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        self.train_op = self._build_train_op()
        self.predict_op = self._build_predict_op()
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        pass

    def _build_train_op(self):
        lrn_rate = 0.01
        opt = tf.train.RMSPropOptimizer(lrn_rate)
        trainable_variables = tf.trainable_variables()
        train_op = opt.minimize(self.cost, var_list=trainable_variables)
        return train_op

    def _build_predict_op(self):
        return self.pred

    def train(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        X = np.reshape(X, (-1))
        X = X[np.newaxis, :]
        y = np.asarray(y, dtype=np.float32)
        self.train_op.run(feed_dict={"state:0":X, "score:0":y}, session=self.sess)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = np.reshape(X, (-1))
        X = X[np.newaxis, :]
        return self.predict_op.eval(feed_dict={"state:0":X}, session=self.sess)[0]
