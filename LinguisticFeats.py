__author__ = 'Nnamdi'

import time
start_time = time.time()
import os, codecs
import numpy
from scipy.sparse import hstack, csr_matrix
from nltk import sent_tokenize, word_tokenize

#Loading pickle file data into numpy array called data
f = open('smoking_1_analytic_data_mapreduce.pkl', 'rb')
data = numpy.load(f)
data = numpy.array(data)
f.close()

#Loading smoking posts_matrix data
posts_matrix = data[0] #11616 rows x 605107 columns
rows = csr_matrix.get_shape(posts_matrix)[0]
users_vector = data[3]
labels_vector = data[4]
keywords_vector = data[2]

#Empty matrix for load columns
loader_matrix = numpy.empty([rows, 8])

def wordCount(fname):
    count = os.system('wc -w %s' % fname)
    return count

def sentPerStatus(fname):
    sent_count = []
    for line in fname:
        sent_count.append(len(sent_tokenize(line)))
    avg = sum(sent_count)/float(len(sent_count))
    return avg

def wordsPerSent(fname):
    word_count = []
    for line in fname:
        for sent in sent_tokenize(line):
            word_count.append(len(word_tokenize(sent)))
    avg = sum(word_count)/float(len(word_count))
    return avg

def punctCount(fname):
    return 1

def lexical_diversity(fname):
    text = fname.read()
    text = word_tokenize(text)
    diversity = len(text) / float(len(set(text)))
    return diversity

def emoticons(fname):
    return 1

def acronyms(fname):
    return 1

def profanities(fname):
    return 1

for user in users_vector:
    os.system('grep -i "{0}{1}" /analytic_store/tmpdata/smoking_1_noUser0.txt > temp.txt'.format('smoking_1_', str(user)))
    fname = codecs.open('/analytic_store/nnamdi/temp.txt', encoding='utf-8')
