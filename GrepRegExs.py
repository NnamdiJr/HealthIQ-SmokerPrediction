__author__ = 'Nnamdi'
import time
start_time = time.time()
import os, codecs
import numpy
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

#Loading pickle file data into numpy array called data
f = open('smoking_1_analytic_data_mapreduce.pkl', 'rb')
data = numpy.load(f)
data = numpy.array(data)
f.close()


#Loading smoking posts_matrix data
posts_matrix = data[1] #11616 rows x 605107 columns
users_vector = data[3]
labels_vector = data[4]


#Empty matrix for load columns
loader_matrix = numpy.empty([csr_matrix.get_shape(posts_matrix)[0], 1])


#List of RegExs
with open('collocation_smoker_regexs.txt','r') as f:
    queries = [l.strip() for l in f]


#Appends each regex column to loader_matrix
for query in queries:
    os.system("grep -i %s /analytic_store/nnamdi/combined_9keywords_text.txt > tmp.txt" % query)
    results = codecs.open('tmp.txt', encoding='utf-8')
    userIDs = [line.split()[0][10:] for line in results]
    dic = {}
    temp_array = numpy.zeros((csr_matrix.get_shape(posts_matrix)[0],1)) #Zeros array
    results.close()

    for user in set(userIDs):
        dic[user] = userIDs.count(user)

    for user in dic.keys():
        if int(user) in users_vector:
            numpy.put(temp_array, [numpy.where(users_vector == int(user))[0][0]], dic[str(user)])
        else:
            continue

    loader_matrix = numpy.hstack((loader_matrix, temp_array))

print "Loader Matrix Shape:", loader_matrix.shape

loader_matrix = csr_matrix(loader_matrix)[:,1:] #Convert loader_matrix to csr matrix and remove first column
combined_matrix = csr_matrix(hstack([posts_matrix, loader_matrix])) #combine loader_matrix with posts_matrix

print "Posts Matrix Shape:", csr_matrix.get_shape(posts_matrix)
print "Combined Matrix Shape:", csr_matrix.get_shape(combined_matrix)

A = posts_matrix
X = combined_matrix
y = labels_vector

holdout_number = (csr_matrix.get_shape(posts_matrix)[0])/5

A_train = A[:holdout_number,:]
A_test = A[holdout_number:,:]
X_train = X[:holdout_number,:]
X_test = X[holdout_number:,:]
y_train = y[:holdout_number]
y_test = y[holdout_number:]

clf01 = LogisticRegression().fit(A_train, y_train)
clf02 = LogisticRegression().fit(X_train, y_train)

model01 = clf01.predict_proba(A_test)
accuracy01 = clf01.score(A_test, y_test)
model02 = clf02.predict_proba(X_test)
accuracy02 = clf02.score(X_test, y_test)


print "Accuracy 01:", accuracy01
print "Accuracy 02:", accuracy02
print "AUC 01:", roc_auc_score(y_test, model01[:,1])
print "AUC 02", roc_auc_score(y_test, model02[:,1])

print("--- %s seconds ---" % (time.time() - start_time))