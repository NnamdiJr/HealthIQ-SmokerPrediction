__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import subprocess
import random
import numpy
from scipy.sparse import hstack, csr_matrix
from nltk import word_tokenize
from nltk import pos_tag
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
loader_matrix = numpy.empty([1, 8])

vrb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adj = ['JJ', 'JJR', 'JJS']
adv = ['RB', 'RBR', 'RBS']
prn = ['PRP', 'PRP$']


def vrb_count(pos_tags, l):
    count = 0
    for tag in pos_tags:
        if tag in vrb:
            count += 1
        continue
    if l == 0:
        avg = 0
    else:
        avg = count/float(l)
    vrb_array = [count, avg]
    return vrb_array


def adj_count(pos_tags, l):
    count = 0
    for tag in pos_tags:
        if tag in adj:
            count += 1
        continue
    if l == 0:
        avg = 0
    else:
        avg = count/float(l)
    adj_array = [count, avg]
    return adj_array


def adv_count(pos_tags, l):
    count = 0
    for tag in pos_tags:
        if tag in adv:
            count += 1
        continue
    if l == 0:
        avg = 0
    else:
        avg = count/float(l)
    adv_array = [count, avg]
    return adv_array


def prn_count(pos_tags, l):
    count = 0
    for tag in pos_tags:
        if tag in prn:
            count += 1
        continue
    if l == 0:
        avg = 0
    else:
        avg = count/float(l)
    prn_array = [count, avg]
    return prn_array


for user in users_vector:
    os.system('fgrep "smoking_1_{0}" users_posts.txt > temp03.txt'.format(str(user)))
    if subprocess.check_output("wc -l temp03.txt", shell=True).strip(' temp03.txt\n') == '':
        lines = 0
    else:
        lines = int(subprocess.check_output("wc -l temp03.txt", shell=True).strip(' temp03.txt\n'))
    temp_file = codecs.open('temp03.txt', encoding='utf-8')
    text = temp_file.read()
    text_words = word_tokenize(text)
    text_pos = [tags[1] for tags in pos_tag(text_words)]
    user_array = numpy.array([vrb_count(text_pos, lines), adj_count(text_pos, lines), adv_count(text_pos, lines),
                               prn_count(text_pos, lines)]).flatten()

    loader_matrix = numpy.vstack((loader_matrix, user_array))
    print("Running Time: %s seconds ||| Current User:" % (time.time() - start_time)), user

loader_matrix = csr_matrix(loader_matrix)[1:, :] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix], format="csr") #combine loader_matrix with posts_matrix

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

    A_train = A[train_indices, :]
    A_test = A[test_indices, :]

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]

    y_train = y[train_indices]
    y_test = y[test_indices]

    clf01 = SGDClassifier(loss="modified_huber").fit(A_train, y_train)
    clf02 = SGDClassifier(loss="modified_huber").fit(X_train, y_train)

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