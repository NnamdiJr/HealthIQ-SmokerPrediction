__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import string
import subprocess
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
loader_matrix = numpy.empty([1, 7])

def word_count(fname):
    count = subprocess.check_output(('wc -w %s' % fname), shell=True)
    count = int(count.strip(' temp.txt\n'))
    return count

def sent_per_status(fname):
    sent_count = []
    with codecs.open(fname, encoding='utf-8') as temp:
        for line in temp:
            sent_count.append(len(sent_tokenize(line)))
    avg = sum(sent_count)/float(len(sent_count))
    temp.close()
    return avg

def words_per_sent(fname):
    word_count = []
    with codecs.open(fname, encoding='utf-8') as temp:
        for line in temp:
            for sent in sent_tokenize(line):
                word_count.append(len(word_tokenize(sent)))
    avg = sum(word_count)/float(len(word_count))
    temp.close()
    return avg

def punct_count(fname):
    count = 0
    temp = codecs.open(fname, encoding='utf-8')
    text = temp.read()
    text = word_tokenize(text)
    for token in text:
        if token in string.punctuation:
            count += 1
        continue
    temp.close()
    return count

def lexical_diversity(fname):
    temp = codecs.open(fname, encoding='utf-8')
    text = temp.read()
    text = word_tokenize(text.lower())
    diversity = len(text) / float(len(set(text)))
    temp.close()
    return diversity

def emoticons(fname):
    count = 0
    with open('emoticons.txt','r') as f:
        emtcns = [l.strip() for l in f]
    with codecs.open(fname, encoding='utf-8') as temp:
        for line in temp:
            for icon in emtcns:
                if icon in line:
                    count += 1
                continue
    temp.close()
    f.close()
    return count

def acronyms(fname):
    count = 0
    temp = codecs.open(fname, encoding='utf-8')
    text = temp.read()
    text = word_tokenize(text.lower())
    with open('acronyms.txt','r') as f:
        acrnms = [l.strip().lower() for l in f]
    for token in text:
        if token in acrnms:
            count += 1
        continue
    temp.close()
    f.close()
    return count

def profanities(fname):
    count = 0
    temp = codecs.open(fname, encoding='utf-8')
    text = temp.read()
    text = word_tokenize(text.lower())
    with open('profanities.txt','r') as f:
        prfnts = [l.strip() for l in f]
    for token in text:
        if token in prfnts:
            count += 1
        continue
    temp.close()
    f.close()
    return count

for user in users_vector[:6]:
    os.system('grep -i "{0}{1}" /analytic_store/tmpdata/smoking_1_noUser0.txt > temp.txt'.format('smoking_1_', str(user)))
    temp_file = 'temp.txt'
    user_array = numpy.array([word_count(temp_file), sent_per_status(temp_file), words_per_sent(temp_file),
                              punct_count(temp_file), lexical_diversity(temp_file), acronyms(temp_file),
                              profanities(temp_file)])

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("--- %s seconds ---" % (time.time() - start_time))

loader_matrix = csr_matrix(loader_matrix)[1:,:] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix],format="csr") #combine loader_matrix with posts_matrix

print "Loader Matrix Shape:", loader_matrix.shape
print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)