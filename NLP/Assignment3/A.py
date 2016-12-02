import main
from sklearn import svm
from sklearn import neighbors
import sys
import nltk
import collections
import codecs

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = collections.defaultdict(set)

    # implement your code here
    for k,v in data.iteritems():
        for (instance_id, left_context, head, right_context, sense_id) in v:
            # left_context
            s[k].update(nltk.word_tokenize(left_context)[-window_size:])

            # right_context
            s[k].update(nltk.word_tokenize(right_context)[:window_size])

    s = {k:list(v) for k,v in s.iteritems()}

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here
    # exclude_chars = [u',', u';', u'"',u"'",u'?',u'/',u')',u'(',u'*',u'&',u'%',u':',u';']
    for (instance_id, left_context, head, right_context, sense_id) in data:
        tks = nltk.word_tokenize(left_context) + nltk.word_tokenize(right_context)
        vectors[instance_id] = [tks.count(w) for w in s]
        labels[instance_id] = sense_id


    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    # implement your code here
    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    X = []
    y = []

    for instance_id, s in X_train.iteritems():
        X.append(s)
        y.append(y_train[instance_id])

    svm_clf.fit(X,y)
    knn_clf.fit(X,y)


    for instance_id, s in X_test.iteritems():
        svm_results.append((instance_id, svm_clf.predict(s)[0]))
        knn_results.append((instance_id, knn_clf.predict(s)[0]))


    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    fo = codecs.open(output_file, encoding='utf-8', mode='w')
    for lexelt, instances in sorted(results.iteritems(), key=lambda d: main.replace_accented(d[0].split('.')[0])):
        for instance_id, sid in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            fo.write(main.replace_accented(lexelt + ' ' + instance_id + ' ' + sid + '\n'))

    fo.close()

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        #print 'Usage: python A.py <language>'
        #sys.exit(0)
        language = 'English'
    else:
        language = sys.argv[1]
    train_file = 'data/' + language + '-train.xml'
    dev_file = 'data/' + language + '-dev.xml'
    train = main.parse_data(train_file)
    test = main.parse_data(dev_file)
    #most_frequent_sense(test, sense_dict,language)
    svm_output = 'SVM-'+language+'.answer'
    knn_output = 'KNN-' + language + '.answer'
    run(train, test, language, knn_output, svm_output)