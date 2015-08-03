__author__ = 'Nnamdi'
import time
start_time = time.time()
import random
import os, codecs
import numpy
from scipy.sparse import hstack, csr_matrix
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
loader_matrix = numpy.empty([rows, 1])

#List of RegExs
with open('collocation_smoker_regexs.txt','r') as f:
    queries = [l.strip() for l in f]

keywords_vector.extend(queries) #Extend keywords list so all columns names accessible

#Appends each regex column to loader_matrix
for query in queries:
    os.system("grep -i %s /analytic_store/nnamdi/combined_9keywords_text.txt > tmp.txt" % query)
    results = codecs.open('tmp.txt', encoding='utf-8')
    userIDs = [line.split()[0][10:] for line in results]
    dic = {}
    temp_array = numpy.zeros((rows,1)) #Zeros array
    results.close()

    for user in set(userIDs):
        dic[user] = userIDs.count(user)

    for user in dic.keys():
        if int(user) in users_vector:
            numpy.put(temp_array, [numpy.where(users_vector == int(user))[0][0]], dic[str(user)])
        else:
            continue

    loader_matrix = numpy.hstack((loader_matrix, temp_array))

loader_matrix = csr_matrix(loader_matrix)[:,1:] #Convert loader_matrix to csr matrix and remove first column
combined_matrix = hstack([posts_matrix, loader_matrix],format="csr") #combine loader_matrix with posts_matrix

print "Loader Matrix Shape:", loader_matrix.shape
print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)

A = posts_matrix
X = combined_matrix
y = labels_vector
del posts_matrix
del combined_matrix

def show_most_informative_features(clf, n=10):
    feature_names = keywords_vector
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

#[index for index, value in enumerate(keywords_vector)]
#i = 0
#while i < 10:
test_indices = numpy.array(random.sample(range(rows), rows/5))
train_indices = numpy.array([num for num in range(rows) if num not in test_indices])

A_train = A[train_indices,:]
A_test = A[test_indices,:]

X_train = X[train_indices,:]
X_test = X[test_indices,:]

y_train = y[train_indices]
y_test = y[test_indices]

clf01 = MultinomialNB().fit(A_train, y_train)
clf02 = MultinomialNB().fit(X_train, y_train)

"""
model01 = clf01.predict_proba(A_test)
accuracy01 = clf01.score(A_test, y_test)
model02 = clf02.predict_proba(X_test)
accuracy02 = clf02.score(X_test, y_test)

print "Accuracy 01:", accuracy01
print "Accuracy 02:", accuracy02
print "AUC 01:", roc_auc_score(y_test, model01[:,1])
print "AUC 02:", roc_auc_score(y_test, model02[:,1])
print("--- %s seconds ---" % (time.time() - start_time))
#i += 1
"""

show_most_informative_features(clf02, n=20)