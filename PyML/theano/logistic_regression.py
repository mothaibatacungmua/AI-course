from PyML.libs import softmax_data_wrapper,shared_dat
import timeit
import numpy as np

import theano
import theano.tensor as T
import cPickle

from layer import LogisticRegression, logistic_error

'''
Code based on: http://deeplearning.net/tutorial/
'''



def SGD(train_data, test_data, validation_data, learning_rate=0.1, epochs=300, batch_size=600):
    X_train = train_data[0]
    y_train = train_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    X_val = validation_data[0]
    y_val = validation_data[1]

    # index to a [mini]batch
    index = T.lscalar()
    X = T.matrix('X')
    y = T.ivector('y')
    classifier = LogisticRegression(npred=28*28, nclass=10)
    classifier.setup(X)

    # Test
    test_model = theano.function(
        inputs=[index],
        outputs=logistic_error(classifier.predict(X), y),
        givens={
            X: X_test[index*batch_size:(index+1)*batch_size],
            y: y_test[index*batch_size:(index+1)*batch_size]
        }
    )


    # Validation
    validate_model = theano.function(
        inputs=[index],
        outputs=logistic_error(classifier.predict(X), y),
        givens={
            X:X_val[index*batch_size:(index+1)*batch_size],
            y:y_val[index*batch_size:(index+1)*batch_size]
        }
    )

    cost = classifier.cost_fn(y)
    grad_W = T.grad(cost, wrt=classifier.W)
    grad_b = T.grad(cost, wrt=classifier.b)

    updates = [
        (classifier.W, classifier.W - learning_rate * grad_W),
        (classifier.b, classifier.b - learning_rate * grad_b)
    ]

    # Train
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            X:X_train[index*batch_size:(index+1)*batch_size],
            y:y_train[index*batch_size:(index+1)*batch_size]
        }
    )

    n_train_batches = X_train.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = X_test.get_value(borrow=True).shape[0] // batch_size
    n_val_batches = X_val.get_value(borrow=True).shape[0] // batch_size

    print('Training the model...')
    #Early-stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < epochs) and (not done_looping):
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

                    print '       Epoch %i, minibatch %i/%i, test error %f' % \
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100)

                    with open('minist_logistic.pkl', 'wb') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print 'Optimization complete with best validation score of %f %%, with test performance %f %%' \
          % (best_validation_loss * 100., test_score * 100.)

    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))


def predict():
    # load trained model
    classifier = cPickle.load(open('minist_logistic.pkl'))

    # predict model
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.predict()
    )

    train, test, validation = softmax_data_wrapper()

    predicted_values = predict_model(test[0].get_value()[:10])
    print("The first 10 examples:")
    print(test[1][:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    train, test, validation = softmax_data_wrapper()
    SGD(shared_dat(train), shared_dat(test), shared_dat(validation))