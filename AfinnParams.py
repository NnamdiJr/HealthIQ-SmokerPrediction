__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import numpy
from scipy.sparse import hstack, csr_matrix
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
loader_matrix = numpy.empty([1, 3])

with open('AFINN-111.txt','r') as f:
        afinns = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f}


def afinn_user_array(tokens):
    count = 0
    aggregate = 0
    for token in tokens:
        if token in afinns.keys():
            count += 1
            aggregate += int(afinns[token])
        continue
    avg = aggregate/float(count)
    array = numpy.array([count, aggregate, avg])
    return array


for user in users_vector[:6]:
    os.system('grep -i "{0}{1}" /analytic_store/tmpdata/smoking_1_noUser0.txt > temp.txt'.format('smoking_1_', str(user)))
    temp_file = codecs.open('temp.txt', encoding='utf-8')
    text = temp_file.read()
    text_words = word_tokenize(text.lower())
    user_array = afinn_user_array(text_words)

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("--- %s seconds ---" % (time.time() - start_time))

loader_matrix = csr_matrix(loader_matrix)[1:,:] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix],format="csr") #combine loader_matrix with posts_matrix