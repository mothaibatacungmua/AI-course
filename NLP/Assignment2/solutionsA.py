import math
import nltk
import time
import collections

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_c = collections.defaultdict(int)
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)

    for s in training_corpus:
        tokens = s.strip().split()

        unigram_tuples = tokens + [STOP_SYMBOL]
        bigram_tuples =  list(nltk.bigrams([START_SYMBOL] + tokens + [STOP_SYMBOL]))
        trigram_tuples = list(nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens + [STOP_SYMBOL]))


        #print unigram_tuples
        for g in unigram_tuples:
            unigram_c[g] += 1

        for g in bigram_tuples:
            bigram_c[g] += 1

        for g in trigram_tuples:
            trigram_c[g] += 1

    count_uni = sum(unigram_c.itervalues())
    unigram_p = {k: math.log(float(v) / count_uni, 2) for k, v in unigram_c.iteritems()}

    unigram_c[START_SYMBOL] = len(training_corpus)
    bigram_p = {k: math.log(float(v) / unigram_c[k[0]], 2) for k, v in bigram_c.iteritems()}

    bigram_c[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {k: math.log(float(v) / bigram_c[k[:2]], 2) for k, v in trigram_c.iteritems()}

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for s in corpus:
        tokens = s.strip().split()
        ss = 0.0
        if n==1:
            mtk = tokens + [STOP_SYMBOL]
        elif n==2:
            mtk = list(nltk.bigrams([START_SYMBOL] + tokens + [STOP_SYMBOL]))
        elif n==3:
            mtk = list(nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens + [STOP_SYMBOL]))

        for token in mtk:
            try:
                ss += ngram_p[token]
            except KeyError:
                ss += MINUS_INFINITY_SENTENCE_LOG_PROB

        scores.append(ss)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for s in corpus:
        lss = 0.0
        tokens = s.strip().split()
        trigram_tuples = list(nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens + [STOP_SYMBOL]))

        for g in trigram_tuples:
            try:
                p3 = trigrams[g]
            except KeyError:
                p3 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p2 = bigrams[g[1:3]]
            except KeyError:
                p2 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p1 = unigrams[g[2]]
            except KeyError:
                p1 = MINUS_INFINITY_SENTENCE_LOG_PROB
            lss += math.log(1.0/3 * (2 ** p3) + 1.0/3 * (2 ** p2) + 1.0/3 * (2 ** p1), 2)

        if lss < MINUS_INFINITY_SENTENCE_LOG_PROB: lss = MINUS_INFINITY_SENTENCE_LOG_PROB
        scores.append(lss)

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
