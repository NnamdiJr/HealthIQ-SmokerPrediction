__author__ = 'Nnamdi'

import time
start_time = time.time()
import codecs
import os
import random
import string
import numpy
from scipy.sparse import hstack, csr_matrix
from nltk import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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

numpy.savetxt('ling_feats_matrix02.txt', loader_matrix[1:, :]) #save loader matrixt to a text file.

loader_matrix = csr_matrix(loader_matrix)[1:,:] #Convert loader_matrix to csr matrix and remove first row
combined_matrix = hstack([posts_matrix, loader_matrix],format="csr") #combine loader_matrix with posts_matrix

print "Loader Matrix Shape:", loader_matrix.shape
print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)


transformer = TfidfTransformer(use_idf=False)
varSelector1 = VarianceThreshold(threshold=0.001)


varSelectorA = SelectKBest(f_classif, k=min(int(11616*0.7), posts_matrix.shape[1]))
varSelectorB = SelectKBest(f_classif, k="all")
varSelectorX = SelectKBest(f_classif, k=min(int(11616*0.7), combined_matrix.shape[1]))


A = varSelector1.fit_transform(posts_matrix)
B = varSelector1.fit_transform(loader_matrix)
X = varSelector1.fit_transform(combined_matrix)
y = labels_vector
del posts_matrix
del combined_matrix


A = transformer.fit_transform(A)
B = transformer.fit_transform(B)
X = transformer.fit_transform(X)


i = 0
while i < 10:
    test_indices = numpy.array(random.sample(range(rows), rows/5))
    train_indices = numpy.array([num for num in range(rows) if num not in test_indices])

    y_train = y[train_indices]
    y_test = y[test_indices]

    A = StandardScaler(with_mean=False).fit_transform(A)
    A_train = A[train_indices, :]
    A_train = varSelectorA.fit_transform(A_train, y_train)
    A_test = A[test_indices, :]
    A_test = varSelectorA.transform(A_test)

    B = StandardScaler(with_mean=False).fit_transform(B)
    B_train = B[train_indices, :]
    B_train = varSelectorB.fit_transform(B_train, y_train)
    B_test = B[test_indices, :]
    B_test = varSelectorB.transform(B_test)

    X = StandardScaler(with_mean=False).fit_transform(X)
    X_train = X[train_indices, :]
    X_train = varSelectorX.fit_transform(X_train, y_train)
    X_test = X[test_indices, :]
    X_test = varSelectorX.transform(X_test)

    clfA = RandomForestClassifier(n_jobs=16, random_state=0).fit(A_train, y_train)
    clfB = RandomForestClassifier(n_jobs=16, random_state=0).fit(B_train, y_train)
    clfX = RandomForestClassifier(n_jobs=16, random_state=0).fit(X_train, y_train)

    modelA = clfA.predict_proba(A_test)
    modelB = clfB.predict_proba(B_test)
    modelX = clfX.predict_proba(X_test)

    print "AUC A:", roc_auc_score(y_test, modelA[:,1])
    print "AUC B:", roc_auc_score(y_test, modelB[:,1])
    print "AUC X:", roc_auc_score(y_test, modelX[:,1])
    print("--- %s seconds ---" % (time.time() - start_time))
    i += 1