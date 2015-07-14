__author__ = 'Nnamdi'

import time
start_time = time.time()
import numpy
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


regex_array = numpy.loadtxt('C:\Users\Nnamdi\Desktop\RegEx_Matrix.txt')
regex_labels = numpy.loadtxt('C:\Users\Nnamdi\Desktop\RegEx_Labels.txt')


X = regex_array
y = regex_labels
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=5)


clf = LogisticRegression().fit(X_train, y_train)
#clf = MultinomialNB().fit(X_train, y_train)

model = clf.predict_proba(X_test)
accuracy = clf.score(X_test, y_test)

print accuracy
print roc_auc_score(y_test, model[:,1])


print("--- %s seconds ---" % (time.time() - start_time))