import A
from sklearn.feature_extraction import DictVectorizer
import sys
import main
import collections
import nltk
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import math
from nltk.corpus import wordnet as wn

# You might change the window size
window_size = 15

def normalize_tokens(tokens, language):
    """Remove punctuation, apply stemming."""
    try:
        stopwords = set(nltk.corpus.stopwords.words(language))
    except IOError:
        stopwords = {}
    return [t for t in tokens if t.isalnum() and t not in stopwords]

def relevance(data):
    '''

    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return:
    '''
    rel = collections.defaultdict(float)
    total = collections.defaultdict(float)
    corpus_tokens = set()
    sen_set = set()

    for (instance_id, left_context, head, right_context, sense_id) in data:
        left_tokens = normalize_tokens(nltk.word_tokenize(left_context.lower()), language)
        right_tokens = normalize_tokens(nltk.word_tokenize(right_context.lower()), language)
        tokens = left_tokens + right_tokens
        corpus_tokens.update(tokens)
        sen_set.update([sense_id])

        for w in (tokens):
            rel[(w, sense_id)] += 1.
            total[w] += 1.

    score = {k:math.log(v/(total[k[0]]-v+1)) for k,v in rel.iteritems()}
    results = {}
    #print sen_set
    for sense_id in sen_set:
        list_key = [(k[0], v) for k,v in score.iteritems() if k[1]==sense_id]
        #print sense_id
        list_key = sorted(list_key, key=lambda v:v[1], reverse=True)
        #print list_key
        results[sense_id] = list_key[0][0]

    return results

# B.1.a,b,c,d
def extract_features(data, language, rel_dict=None):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}
    exclude_chars = [u',', u';', u'"', u"'", u'?', u'/', u')', u'(', u'*', u'&', u'%', u':', u';']
    corpus_tokens = set()

    try:
        stemmer = nltk.stem.snowball.SnowballStemmer(language)
    except ValueError:
        stemmer = nltk.stem.lancaster.LancasterStemmer()

    # implement your code here
    for (instance_id, left_context, head, right_context, sense_id) in data:
        left_tokens = [stemmer.stem(t) for t in
                       normalize_tokens(nltk.word_tokenize(left_context.lower()), language)][-window_size:]

        right_tokens = [stemmer.stem(t) for t in
                        normalize_tokens(nltk.word_tokenize(right_context.lower()), language)][:window_size]

        corpus_tokens.update(left_tokens + right_tokens)
        labels[instance_id] = sense_id
        features[instance_id] = collections.defaultdict(float)


    for (instance_id, left_context, head, right_context, sense_id) in data:
        left_tokens = normalize_tokens(nltk.word_tokenize(left_context.lower()), language)
        right_tokens = normalize_tokens(nltk.word_tokenize(right_context.lower()), language)
        tks =  [stemmer.stem(t) for t in left_tokens] + [stemmer.stem(t) for t in right_tokens]

        for w in corpus_tokens:
            if w not in exclude_chars:
                features[instance_id][u'TK_'+w] = tks.count(w)*1.0

        for l in left_tokens[-window_size:]:
            if l not in exclude_chars:
                features[instance_id][u'LEFT_' + l] += 1.0

                #adding hypernyms, synomous
                if language == 'en':
                    for synset in wn.synsets(l)[:2]:
                        for lemma in synset.lemmas()[:3]:
                            if lemma.name() != l:
                                features[instance_id][u'SYN_LEFT_' + lemma.name()] += 1.0

                        hypernyms = synset.hypernyms()
                        #print hypernyms
                        for h in hypernyms[:2]:
                            lemma = h.lemmas()[0]
                            features[instance_id][u'HYPER_LEFT_' + lemma.name()] = 1.0

        for r in right_tokens[:window_size]:
            if r not in exclude_chars:
                features[instance_id][u'RIGHT_' + r] += 1.0

                if language == 'en':
                    for synset in wn.synsets(r)[:2]:
                        for lemma in synset.lemmas()[:3]:
                            if lemma.name() != r:
                                features[instance_id][u'SYN_RIGHT_' + lemma.name()] += 1.

                        hypernyms = synset.hypernyms()
                        #print hypernyms
                        for h in hypernyms[:2]:
                            lemma = h.lemmas()[0]
                            features[instance_id][u'HYPER_RIGHT_' + lemma.name()] = 1.0

        #relevance score
        #print rel_dict
        for sense_id, w in rel_dict.iteritems():
            if stemmer.stem(w) in tks:
                features[instance_id][u'REL_' + w + '_' + sense_id] = 1

        #print features[instance_id]
    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    clf = svm.LinearSVC()


    '''
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=500,
        learning_rate=0.1)
    '''

    # implement your code here

    X = []
    y = []

    for instance_id, s in X_train.iteritems():
        X.append(s)
        y.append(y_train[instance_id])

    clf.fit(X, y)

    for instance_id, s in X_test.iteritems():
        results.append((instance_id, clf.predict(s)[0]))

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    if language == 'English': language = 'en'
    if language == 'Spanish': language = 'spa'
    if language == 'Catalan': language = 'cat'

    for lexelt in train:
        rel_dict = relevance(train[lexelt])
        train_features, y_train = extract_features(train[lexelt], language, rel_dict=rel_dict)
        test_features, _ = extract_features(test[lexelt], language, rel_dict=rel_dict)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)

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
    output = 'Best-'+language+'.answer'
    run(train, test, language, output)