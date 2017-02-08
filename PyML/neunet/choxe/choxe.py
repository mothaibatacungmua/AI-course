import cPickle
import numpy
from PIL import Image
import os
import numpy.matlib

X = []
def get_img_files(fdir):
    count = 0
    for filename in os.listdir(fdir):
        if filename.endswith(".jpg"):
            img = Image.open(open(fdir + filename))
            # dimensions are (height, width, channel)
            resized_img = img.resize((256, 256), Image.ANTIALIAS)
            resized_img = numpy.asarray(resized_img, dtype='float32') / 256.
            X.append(resized_img)
            count += 1
    return count


def get_training_data():
    global X
    X = []
    count = get_img_files("./data/dong-co/")
    y1 = numpy.array([1, 0, 0])
    y1 = numpy.matlib.repmat(y1, count, 1)

    count = get_img_files("./data/ngoai-that/")
    y2 = numpy.array([0, 1, 0])
    y2 = numpy.matlib.repmat(y2, count, 1)

    count = get_img_files("./data/noi-that/")
    y3 = numpy.array([0, 0, 1])
    y3 = numpy.matlib.repmat(y3, count, 1)

    #print len(X)
    X = numpy.asarray(X, dtype='float32')

    y = numpy.concatenate((y1, y2, y3), axis=0)

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
        with open(DATA_FILE, 'rb') as pickle_file:
            return cPickle.load(pickle_file)
    else:
        X_train, y_train = get_training_data()
        X_test, y_test = get_testing_data()
        data = (X_train, y_train, X_test, y_test)
        with open(DATA_FILE, 'wb') as pickle_file:
            cPickle.dump(data, pickle_file, protocol=2)
        return data