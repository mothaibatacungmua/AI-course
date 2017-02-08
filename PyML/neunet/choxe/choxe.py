import cPickle
import numpy
from PIL import Image
import os
import numpy.matlib

def get_img_files(fdir):
    X = []
    for filename in os.listdir(fdir):
        if filename.endswith(".jpg"):
            print filename
            img = Image.open(open(fdir + filename))
            # dimensions are (height, width, channel)
            img = numpy.asarray(img, dtype='float32') / 256.
            resized_img = img.resize((256, 256), Image.ANTIALIAS)
            #img = img.transpose(2, 0, 1)
            X.append(resized_img)
    return X


def get_training_data():
    X1 = get_img_files("./data/dong-co/")
    y1 = numpy.array([1, 0, 0])
    y1 = numpy.matlib.repmat(y1, len(X1), 1)

    X2 = get_img_files("./data/ngoai-that")
    y2 = numpy.array([0, 1, 0])
    y2 = numpy.matlib.repmat(y2, len(X2), 1)

    X3 = get_img_files("./data/noi-that")
    y3 = numpy.array([0, 0, 1])
    y3 = numpy.matlib.repmat(y3, len(X3), 1)

    X = X1 + X2 + X3
    X = numpy.reshape(X, (len(X), 256, 256, 3))
    y = numpy.concatenate(y1, y2, y3, axis=0)

    #shuffle training data
    perm = numpy.random.permutation(y.shape[0])

    X = X[perm]
    y = y[perm]

    return (X, y)

def get_testing_data():
    return (None, None)
    pass

DATA_FILE = "choxe.pkl"
def load_data():
    if os.path.isfile(DATA_FILE):
        return cPickle.load(DATA_FILE)
    else:
        X_train, y_train = get_training_data()
        X_test, y_test = get_testing_data()
        data = (X_train, y_train, X_test, y_test)
        cPickle.dump(data, DATA_FILE)
        return data