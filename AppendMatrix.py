__author__ = 'Nnamdi'

import time
start_time = time.time()
from scipy import sparse
from scipy.sparse import vstack, hstack, csr_matrix
import numpy
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

print("--- %s seconds ---" % (time.time() - start_time))

#Loading pickle file data into numpy array called data
f = open('C:\Users\Nnamdi\Desktop\HealthIQ\Smoker Analysis\smoking_1_analytic_data_mapreduce.pkl', 'rb')
data = numpy.load(f)
data = numpy.array(data)
f.close()

#Loading both matrices
posts_matrix = data[1] #11616 rows x 605107 columns
regex_matrix = sparse.csr_matrix(numpy.loadtxt('C:\Users\Nnamdi\Desktop\RegEx_Matrix.txt'))
labels_vector = data[4]

#Loading both sets of userIDs
riskFactor_userIDs = data[3] #Vector containing risk factor matrix userIDs
regex_userIDs = numpy.loadtxt('C:\Users\Nnamdi\Desktop\RegEx_UserIDs.txt') #Vector containing regex matrix userIDs


def appendMatrix(matrix01, matrix02):
    #Column-wise appends matrix02 columns to matrix01, where user indices are unaligned. Returns new matrix01.
    hold_matrix = csr_matrix((1, csr_matrix.get_shape(matrix02)[1])) #Empty matrix with 1 row

    for user in riskFactor_userIDs:
        if user in regex_userIDs:
            hold_matrix = vstack([hold_matrix, matrix02[numpy.where(regex_userIDs==user)[0][0],:]])
            continue
        hold_matrix = vstack([hold_matrix, numpy.zeros(csr_matrix.get_shape(matrix02)[1])])

    hold_matrix = csr_matrix(hold_matrix)[1:,:] #convert to csr matrix and remove first row
    matrix01 = csr_matrix(hstack([matrix01, hold_matrix]))
    return matrix01

X = posts_matrix[:400,:]
#X = appendMatrix(posts_matrix, regex_matrix)
y = labels_vector[:400]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(probability=True,kernel='linear').fit(X_train, y_train)
#clf = MultinomialNB().fit(X_train, y_train)

model = clf.predict_proba(X_test)
accuracy = clf.score(X_test, y_test)

print roc_auc_score(y_test, model[:,1])

print("--- %s seconds ---" % (time.time() - start_time))