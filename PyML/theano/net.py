import timeit
import theano
import theano.tensor as T
from const import LAYER_TYPES
import numpy as np


class MLP(object):
    def __init__(self, inp, target, error_fn, L1_reg=0.00, L2_reg=0.0001):
        self.L1 = 0
        self.L2 = 0
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.layers = []
        self.params = []
        self.calc_cost = None
        self.input = inp
        self.error_fn = error_fn
        self.target = target

    def convert_output_to_input(self, prev_layer_output, prev_layer, next_layer):
        if prev_layer is None:
            if type(next_layer) != LAYER_TYPES['ConvPoolLayer']:
                return prev_layer_output

            return prev_layer_output.reshape(next_layer.image_shape)

        if type(prev_layer) == type(next_layer):
            return prev_layer_output

        if type(prev_layer) != LAYER_TYPES['ConvPoolLayer'] and \
           type(next_layer) != LAYER_TYPES['ConvPoolLayer']:
            return prev_layer_output

        # assume that ignore_border=True
        if type(prev_layer) == LAYER_TYPES['ConvPoolLayer']:
            # the HiddenLayer being fully-connected, it operates on 2D matrices of
            # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
            return prev_layer_output.flatten(2)

        pass

    def add_layer(self, layer):
        self.layers.append(layer)

        if type(layer) != LAYER_TYPES['ConvPoolLayer']:
            self.L1 += abs(layer.W).sum()
            self.L2 += (layer.W ** 2).sum()

        self.params += layer.params

        if len(self.layers) == 1:
            layer.setup(self.convert_output_to_input(self.input, None, layer))
            return

        prev_layer = self.layers[-2]
        layer.setup(self.convert_output_to_input(prev_layer.output, prev_layer, layer))
        return

    # feedforward phase in the testing process
    def _test_f(self, X, l_i):
        if l_i == (len(self.layers) - 2):
            return self.layers[l_i].feedforward(X)

        c_X = X
        if l_i == 0:
            c_X = self.convert_output_to_input(X, None, self.layers[0])

        n_X = self.convert_output_to_input(
            self.layers[l_i].feedforward(c_X),
            self.layers[l_i],
            self.layers[l_i+1])

        return self._test_f(n_X, l_i + 1)

    # feedforward phase in the training process
    def _train_f(self, X, l_i):
        if l_i == (len(self.layers) - 2):
            return self.layers[l_i].calc_output(X)

        c_X = X
        if l_i == 0:
            c_X = self.convert_output_to_input(X, None, self.layers[0])

        n_X = self.convert_output_to_input(
            self.layers[l_i].calc_output(c_X),
            self.layers[l_i],
            self.layers[l_i + 1])

        return self._train_f(n_X, l_i + 1)


    def cost_fn(self, X, y):
        last_inp = self._train_f(X, 0)
        return self.layers[-1].cost_fn(last_inp, y) + self.L1_reg * self.L1 + self.L2_reg * self.L2

    def predict(self, X):
        last_inp = self._test_f(X, 0)
        return self.layers[-1].predict(last_inp)

    def gen_test_model(self, X, y, batch_size):
        minibatch_index = T.lscalar()

        return theano.function(
            inputs=[minibatch_index],
            outputs=self.error_fn(self.predict(X), y),
            givens={
                X: X[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                y: y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            }
        )

    def gen_validate_model(self, X, y, batch_size):
        minibatch_index = T.lscalar()

        return theano.function(
            inputs=[minibatch_index],
            outputs=self.error_fn(self.predict(X), y),
            givens={
                X: X[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                y: y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            }
        )

    def gen_train_model(self, X, y, batch_size, learning_rate, optimization='SGD'):
        cost = self.cost_fn(X, y)
        grad = [T.grad(cost, param) for param in self.params]

        # updates
        updates = [
            (param, param - learning_rate * grad)
            for param, grad in zip(self.params, grad)
        ]

        # Train model
        minibatch_index = T.lscalar()

        return theano.function(
            inputs=[minibatch_index],
            outputs=cost,
            updates=updates,
            givens={
                X: X[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                y: y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            }
        )


def train_mlp(train_data, test_data, validation_data, neunet, learning_rate=0.1, batch_size=30, epochs=500, optimization='SGD'):
    X_train = train_data[0]
    y_train = train_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    X_val = validation_data[0]
    y_val = validation_data[1]


    # Test model
    test_model = neunet.gen_test_model(X_test, y_test, batch_size)

    # Validation model
    validate_model = neunet.gen_validate_model(X_val, y_val, batch_size)

    # Train model
    train_model = neunet.gen_train_model(X_train, y_train, batch_size, learning_rate, optimization=optimization)

    n_train_batches = X_train.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = X_test.get_value(borrow=True).shape[0] // batch_size
    n_val_batches = X_val.get_value(borrow=True).shape[0] // batch_size

    print('Training the model...')
    # Early-stopping parameters
    patience = 20000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_model = None
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    n = X_train.get_value(borrow=True).shape[0]
    while (epoch < epochs) and (not done_looping):
        #shuffe train data
        rand_perm = np.random.permutation(n)
        X_train = X_train[rand_perm,:]
        y_train = y_train[rand_perm]

        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_train_lost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            # Monitor validation error
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_val_batches)]
                validation_cost = np.mean(validation_losses)

                print 'Epoch %i, minibatch %i/%i, validation error %f' % \
                      (epoch, minibatch_index + 1, n_train_batches, validation_cost * 100)


                # Early stopping
                if validation_cost < best_validation_loss:
                    if validation_cost < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                        best_validation_loss = validation_cost

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print '     Epoch %i, minibatch %i/%i, test error %f' % \
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100)

                    #TODO: save MLP params

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print 'Optimization complete with best validation score of %f %%, with test performance %f %%' \
          % (best_validation_loss * 100., test_score * 100.)

    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))