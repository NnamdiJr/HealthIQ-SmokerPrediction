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
from sklearn.naive_bayes import MultinomialNB
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
loader_matrix = numpy.empty([1, 9])

with open('h4lvd_pstv.txt', 'r') as f:
        positive = [line.strip() for line in f]
with open('h4lvd_ngtv.txt', 'r') as f:
        negative = [line.strip() for line in f]
with open('h4lvd_strng.txt', 'r') as f:
        strong = [line.strip() for line in f]
with open('h4lvd_weak.txt', 'r') as f:
        weak = [line.strip() for line in f]
with open('h4lvd_actv.txt', 'r') as f:
        active = [line.strip() for line in f]
with open('h4lvd_psv.txt', 'r') as f:
        passive = [line.strip() for line in f]


def h4lvd_array(tokens, lines):
    pstv_count = 0
    ngtv_count = 0
    strng_count = 0
    weak_count = 0
    actv_count = 0
    psv_count = 0
    for token in tokens:
        if token in positive:
            pstv_count += 1
        if token in negative:
            ngtv_count += 1
        if token in strong:
            strng_count += 1
        if token in weak:
            weak_count += 1
        if token in active:
            actv_count += 1
        if token in passive:
            psv_count += 1
        continue
    if ngtv_count == 0:
        pstv_ngtv_ratio = 0
    else:
        pstv_ngtv_ratio = pstv_count/ngtv_count
    if weak_count == 0:
        strng_weak_ratio = 0
    else:
        strng_weak_ratio = strng_count/weak_count
    if psv_count == 0:
        actv_psv_ratio = 0
    else:
        actv_psv_ratio = actv_count/psv_count

    return numpy.array([pstv_count/float(lines), ngtv_count/float(lines), strng_count/float(lines),
                        weak_count/float(lines), actv_count/float(lines), psv_count/float(lines), pstv_ngtv_ratio,
                        strng_weak_ratio, actv_psv_ratio])

for user in users_vector:
    os.system('fgrep "smoking_1_{0}" users_posts.txt > temp04.txt'.format(str(user)))
    if subprocess.check_output("wc -l temp04.txt", shell=True).strip(' temp04.txt\n') == '':
        user_array = numpy.zeros(9)
        loader_matrix = numpy.vstack((loader_matrix, user_array))
        print("Running Time: %s seconds ||| Current User:" % (time.time() - start_time)), user
        continue
    else:
        posts = int(subprocess.check_output("wc -l temp04.txt", shell=True).strip(' temp04.txt\n'))
        temp_file = codecs.open('temp04.txt', encoding='utf-8')
        text = temp_file.read()
        text_words = word_tokenize(text.upper())
        user_array = h4lvd_array(text_words, posts)

        loader_matrix = numpy.vstack((loader_matrix, user_array))
        print("Running Time: %s seconds ||| Current User:" % (time.time() - start_time)), user

numpy.savetxt('h4lvd_matrix.txt', loader_matrix[1:, :]) #save loader matrixt to a text file.

loader_matrix = csr_matrix(loader_matrix)[1:, :] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix], format="csr") #combine loader_matrix with posts_matrix

print "Loader Matrix Shape:", loader_matrix.shape
print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)

A = posts_matrix
B = loader_matrix
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

    B_train = B[train_indices, :]
    B_test = B[test_indices, :]

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]

    y_train = y[train_indices]
    y_test = y[test_indices]

    clfA = MultinomialNB().fit(A_train, y_train)
    clfB = MultinomialNB().fit(B_train, y_train)
    clfX = MultinomialNB().fit(X_train, y_train)

    modelA = clfA.predict_proba(A_test)
    modelB = clfB.predict_proba(B_test)
    modelX = clfX.predict_proba(X_test)

    print "AUC A:", roc_auc_score(y_test, modelA[:,1])
    print "AUC B:", roc_auc_score(y_test, modelB[:,1])
    print "AUC X:", roc_auc_score(y_test, modelX[:,1])
    print("--- %s seconds ---" % (time.time() - start_time))
    i += 1