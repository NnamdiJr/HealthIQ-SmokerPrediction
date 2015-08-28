__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import numpy
from scipy.sparse import csr_matrix
from nltk import word_tokenize

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
loader_matrix = numpy.empty([1, 14])

with open('AFINN-111.txt','r') as f:
        afinns = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f}


def afinn_user_array(tokens):
    count = 0
    aggregate = 0
    ints = []
    for token in tokens:
        if token in afinns.keys():
            count += 1
            aggregate += int(afinns[token])
            ints.append(int(afinns[token]))
    if count == 0:
        avg = 0
    else:
        avg = aggregate/float(count)
    array = numpy.array([count, aggregate, avg, ints.count(-5), ints.count(-4), ints.count(-3), ints.count(-2),
                         ints.count(-1), ints.count(0), ints.count(1), ints.count(2), ints.count(3), ints.count(4),
                         ints.count(5)])
    return array


for user in users_vector:
    os.system('fgrep "smoking_1_{0}" users_posts.txt > temp01.txt'.format(str(user)))
    temp_file = codecs.open('temp01.txt', encoding='utf-8')
    text = temp_file.read()
    text_words = word_tokenize(text.lower())
    user_array = afinn_user_array(text_words)

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("Running Time: %s seconds ||| Current User:" % (time.time() - start_time)), user

numpy.savetxt('afinn_params_matrix02.txt', loader_matrix[1:, :]) #save loader matrix to a text file.