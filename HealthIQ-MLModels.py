__author__ = 'Nnamdi'

import time
start_time = time.time()
import numpy
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack, csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

#Opens pickled file and loads data into numpy array called data
f=open('C:\Users\Nnamdi\Desktop\HealthIQ\Smoker Analysis\smoking_analytic_data_1.pkl', 'rb')
data = numpy.load(f)
data = numpy.array(data)
f.close()


#Rename data elements by their corresponding properties
likes_matrix = data[0]
posts_matrix = data[1]
keywords_vector = data[2] #size is 587002
users_vector = data[3] #size is 2985
labels_vector = data[4]

print len(labels_vector)

"""
#Reads hand-picked keywords from text file
lst = []
lst_file = open("C:\Users\Nnamdi\Desktop\HealthIQ\Data Generated Keyword Processing\Smoking\smoking_top3000_keywords_unprocessed_123Top200.txt", "r")
for line in lst_file:
    lst.append(line.strip('\n'))
lst_file.close()


#Creates a new matrix, X, consisting of just the columns of keyword indices
X = csr_matrix((2985,1)) #Creates empty CSR matrix
for keyword in keywords_vector:
    if keyword in lst:
        Z = posts_matrix[:,keywords_vector.index(keyword)]
        X = hstack([X,Z])
    continue


y = labels_vector
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

clf = LogisticRegression().fit(X_train, y_train)
#clf = MultinomialNB().fit(X_train, y_train)

model = clf.predict_proba(X_test)
accuracy = clf.score(X_test, y_test)

print accuracy
print roc_auc_score(y_test, model[:,1])
"""

print("--- %s seconds ---" % (time.time() - start_time))