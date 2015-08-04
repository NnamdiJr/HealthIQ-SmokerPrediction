__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import string
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


def word_count(tokens):
    count = len(tokens)
    return count


def sent_per_status(file_text):
    sent_count = []
    for line in file_text:
        sent_count.append(len(sent_tokenize(line)))
    avg = sum(sent_count)/float(len(sent_count))
    return avg


def words_per_sent(sents):
    word_counts = []
    for sent in sents:
        word_counts.append(len(word_tokenize(sent)))
    avg = sum(word_counts)/float(len(word_counts))
    return avg


def punct_count(tokens):
    count = 0
    for token in tokens:
        if token in string.punctuation:
            count += 1
        continue
    return count


def lexical_diversity(tokens):
    diversity = len(tokens) / float(len(set(tokens)))
    return diversity


def emoticons(file_text):
    count = 0
    with open('emoticons.txt', 'r') as f:
        emtcns = [l.strip() for l in f]
    for line in file_text:
        for icon in emtcns:
            if icon in line:
                count += 1
            continue
    f.close()
    return count


def acronyms(tokens):
    count = 0
    with open('acronyms.txt','r') as f:
        acrnms = [l.strip().lower() for l in f]
    for token in tokens:
        if token in acrnms:
            count += 1
        continue
    f.close()
    return count


def profanities(tokens):
    count = 0
    with open('profanities.txt', 'r') as f:
        prfnts = [l.strip() for l in f]
    for token in tokens:
        if token in prfnts:
            count += 1
        continue
    f.close()
    return count

for user in users_vector[:6]:
    os.system('grep -i "{0}{1}" /analytic_store/tmpdata/smoking_1_noUser0.txt > temp.txt'.format('smoking_1_', str(user)))
    temp_file = codecs.open('temp.txt', encoding='utf-8')
    text = temp_file.read()
    text_sents = sent_tokenize(text)
    text_words = word_tokenize(text.lower())
    user_array = numpy.array([word_count(text_words), sent_per_status(text), words_per_sent(text_sents),
                              punct_count(text_words), lexical_diversity(text_words), acronyms(text_words),
                              profanities(text_words)])

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("--- %s seconds ---" % (time.time() - start_time))

loader_matrix = csr_matrix(loader_matrix)[1:,:] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix],format="csr") #combine loader_matrix with posts_matrix

print "Loader Matrix Shape:", loader_matrix.shape
print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)