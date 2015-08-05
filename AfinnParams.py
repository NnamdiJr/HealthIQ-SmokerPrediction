__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import random
import numpy
from scipy.sparse import hstack, csr_matrix
from nltk import word_tokenize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

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
    if count == 0:
        avg = 0
    else:
        avg = aggregate/float(count)
    array = numpy.array([count, aggregate, avg])
    return array


for user in users_vector:
    os.system('fgrep "smoking_1_{0}" users_posts.txt > temp01.txt'.format(str(user)))
    temp_file = codecs.open('temp01.txt', encoding='utf-8')
    text = temp_file.read()
    text_words = word_tokenize(text.lower())
    user_array = afinn_user_array(text_words)

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("Running Time: %s seconds ||| Current User:" % (time.time() - start_time)), user


loader_matrix = csr_matrix(loader_matrix)[1:,:] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix],format="csr") #combine loader_matrix with posts_matrix

print "Loader Matrix Shape:", loader_matrix.shape
print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)

A = posts_matrix
X = combined_matrix
y = labels_vector
del posts_matrix
del combined_matrix

i = 0
while i < 10:
    test_indices = numpy.array(random.sample(range(rows), rows/5))
    train_indices = numpy.array([num for num in range(rows) if num not in test_indices])

    A_train = A[train_indices,:]
    A_test = A[test_indices,:]

    X_train = X[train_indices,:]
    X_test = X[test_indices,:]

    y_train = y[train_indices]
    y_test = y[test_indices]

    clf01 = SGDClassifier().fit(A_train, y_train)
    clf02 = SGDClassifier().fit(X_train, y_train)

    model01 = clf01.predict_proba(A_test)
    accuracy01 = clf01.score(A_test, y_test)
    model02 = clf02.predict_proba(X_test)
    accuracy02 = clf02.score(X_test, y_test)

    print "Accuracy 01:", accuracy01
    print "Accuracy 02:", accuracy02
    print "AUC 01:", roc_auc_score(y_test, model01[:,1])
    print "AUC 02:", roc_auc_score(y_test, model02[:,1])
    print("--- %s seconds ---" % (time.time() - start_time))
    i += 1