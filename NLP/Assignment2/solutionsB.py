import sys
import nltk
import math
import time
import collections
import itertools

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for s in brown_train:
        tokens = s.strip().split()
        s_words = [START_SYMBOL]*2
        s_tags = [START_SYMBOL]*2

        for wtag in tokens:
            lidx = wtag.rindex(u'/')
            word = wtag[:lidx]
            tag = wtag[lidx+1:]
            s_words.append(word)
            s_tags.append(tag)

        s_words.append(STOP_SYMBOL)
        s_tags.append(STOP_SYMBOL)

        brown_words.append(s_words)
        brown_tags.append(s_tags)
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    #print brown_tags[0]
    #q_values = {}
    #unigram_c = collections.defaultdict(int)
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)

    for stags in brown_tags:
        unigram_tuples = stags
        bigram_tuples =  list(nltk.bigrams(stags))
        trigram_tuples = list(nltk.trigrams(stags))


        #print unigram_tuples
        #for g in unigram_tuples:
            #unigram_c[g] += 1

        for g in bigram_tuples:
            bigram_c[g] += 1

        for g in trigram_tuples:
            trigram_c[g] += 1

    bigram_c[(START_SYMBOL, START_SYMBOL)] = len(brown_tags)
    q_values = {k: math.log(float(v) / bigram_c[k[:2]], 2) for k, v in trigram_c.iteritems()}

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    #print brown_words[0]
    known_words = set([])
    com_words = collections.defaultdict(int)
    for swords in brown_words:
        for w in swords:
            com_words[w] += 1

    for k,v in com_words.iteritems():
        if v > 5:
            known_words.add(k)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for swords in brown_words:
        newsw = swords[:]
        for i in range(len(newsw)):
            if not newsw[i] in known_words:
                newsw[i] = RARE_SYMBOL

        brown_words_rare.append(newsw)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    wordtags = collections.defaultdict(int)
    tagdict = collections.defaultdict(int)

    for i in range(len(brown_words_rare)):
        for w,t in zip(brown_words_rare[i], brown_tags[i]):
            wordtags[(w,t)] += 1
            tagdict[t] += 1
            taglist.add(t)

    e_values = {k:math.log(v*1.0, 2) - math.log(tagdict[k[1]]*1.0, 2) for k,v in wordtags.iteritems()}

    taglist = set([k for k in tagdict.keys()])
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!

def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    f = collections.defaultdict(float)
    back_pointers= {}
    back_pointers[(-1, START_SYMBOL, START_SYMBOL)] = START_SYMBOL
    f[(-1, START_SYMBOL, START_SYMBOL)] = 0.0

    for sen_words in brown_dev_words:
        tokens = [w if w in known_words else RARE_SYMBOL for w in sen_words]

        for w in taglist:
            word_tag = (tokens[0], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            f[(0, START_SYMBOL, w)] = f[(-1, START_SYMBOL, START_SYMBOL)] + \
                                       q_values.get(trigram, LOG_PROB_OF_ZERO) + \
                                       e_values.get(word_tag, LOG_PROB_OF_ZERO)
            back_pointers[(0, START_SYMBOL, w)] = START_SYMBOL

        for w in taglist:
            for u in taglist:
                word_tag = (tokens[1], u)
                trigram = (START_SYMBOL, w, u)
                f[(1, w, u)] = f.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + \
                                q_values.get(trigram, LOG_PROB_OF_ZERO) + \
                                e_values.get(word_tag, LOG_PROB_OF_ZERO)
                back_pointers[(1, w, u)] = START_SYMBOL

        for k in range(2, len(tokens)):
            for tt in taglist:
                for t in taglist:
                    max_prob = float('-Inf')
                    max_tag = ''
                    for ttt in taglist:
                        score = f.get((k - 1, ttt, tt), LOG_PROB_OF_ZERO) + \
                                q_values.get((ttt, tt, t), LOG_PROB_OF_ZERO) + \
                                e_values.get((tokens[k], t), LOG_PROB_OF_ZERO)
                        if (score > max_prob):
                            max_prob = score
                            max_tag = ttt
                    back_pointers[(k, tt, t)] = max_tag
                    f[(k, tt, t)] = max_prob

        max_log_prob = float('-Inf')
        ttt_max, tt_max = None, None
        for (ttt, tt) in itertools.product(taglist, taglist):
            log_prob = q_values.get((ttt, tt, STOP_SYMBOL), LOG_PROB_OF_ZERO) + \
                      f.get((len(tokens) - 1, ttt, tt), LOG_PROB_OF_ZERO)
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                ttt_max = ttt
                tt_max = tt

        sen_tagged = []
        sen_tagged.append(tt_max)
        sen_tagged.append(ttt_max)


        for count, i in enumerate(range(len(tokens)-3,-1,-1)):
            sen_tagged.append(back_pointers[(i+2, sen_tagged[count+1], sen_tagged[count])])

        sen_tagged.reverse()
        wtagged = []
        for wt in zip(sen_words, sen_tagged):
            wtagged.append(wt[0]+'/'+wt[1])
        wtagged.append('\n')

        tagged.append(' '.join(wtagged))

    return tagged


# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for s in brown_dev_words:
        tagged.append(' '.join([word + '/' + tag for word, tag in trigram_tagger.tag(s)]) + '\n')

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
