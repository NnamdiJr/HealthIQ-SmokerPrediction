__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import string
import numpy
from scipy.sparse import csr_matrix
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
    count = len([token for token in tokens if token not in string.punctuation])
    return count


def sent_per_status(file_text):
    content = file_text.readlines()
    sent_count = [len(sent_tokenize(line)) for line in content]
    if len(sent_count) == 0:
        avg = 0
    else:
        avg = sum(sent_count)/float(len(sent_count))
    return avg


def words_per_sent(sents):
    word_counts = []
    for sent in sents:
        count = len([token for token in word_tokenize(sent) if token not in string.punctuation])
        word_counts.append(count)
    if len(word_counts) == 0:
        avg = 0
    else:
        avg = sum(word_counts)/float(len(word_counts))
    return avg


def punct_count(tokens):
    count = 0
    for token in tokens:
        if token in string.punctuation:
            count += 1
    return count


def lexical_diversity(tokens):
    tokens = [token for token in tokens if token not in string.punctuation]
    if len(tokens) == 0:
        diversity = 0
    else:
        diversity = len(tokens) / float(len(set(tokens)))
    return diversity


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

#Creates feature matrix
for user in users_vector:
    os.system('fgrep "smoking_1_{0}" users_posts.txt > temp02.txt'.format(str(user)))
    temp_file = codecs.open('temp02.txt', encoding='utf-8')
    text = temp_file.read()
    text_sents = sent_tokenize(text)
    text_words = word_tokenize(text.lower())
    user_array = numpy.array([word_count(text_words), sent_per_status(temp_file), words_per_sent(text_sents),
                              punct_count(text_words), lexical_diversity(text_words), acronyms(text_words),
                              profanities(text_words)])

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("Running Time: %s seconds ||| Current User:" % (time.time() - start_time)), user

numpy.savetxt('ling_feats_matrix02.txt', loader_matrix[1:, :]) #save loader matrix to a text file.