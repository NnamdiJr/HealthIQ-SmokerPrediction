__author__ = 'Nnamdi'

import time
start_time = time.time()
from scipy import sparse
from scipy.sparse import vstack, hstack, csr_matrix
import numpy

#Loading pickle file data into numpy array called data
f = open('C:\Users\Nnamdi\Desktop\HealthIQ\Smoker Analysis\smoking_analytic_data_1.pkl', 'rb')
data = numpy.load(f)
data = numpy.array(data)
f.close()

#Loading both matrices
posts_matrix = data[1]
regex_matrix = sparse.csr_matrix(numpy.loadtxt('C:\Users\Nnamdi\Desktop\RegEx_Matrix.txt'))

#Loading both sets of userIDs
riskFactor_userIDs = data[3] #Vector containing risk factor matrix userIDs
regex_userIDs = numpy.loadtxt('C:\Users\Nnamdi\Desktop\RegEx_UserIDs.txt') #Vector containing regex matrix userIDs


def appendMatrix(matrix01, matrix02):
    """Column-wise appends matrix02 to matrix01, where user indices are unaligned. Returns new matrix01."""
    hold_matrix = csr_matrix((1, csr_matrix.get_shape(matrix02)[1])) #Empty matrix with 1 row

    for user in riskFactor_userIDs:
        if user in regex_userIDs:
            hold_matrix = vstack([hold_matrix, matrix02[numpy.where(regex_userIDs==user)[0][0],:]])
            continue
        hold_matrix = vstack([hold_matrix, numpy.zeros(csr_matrix.get_shape(matrix02)[1])])

    hold_matrix = csr_matrix(hold_matrix)[1:,:] #convert to csr matrix and remove first row
    matrix01 = csr_matrix(hstack([matrix01, hold_matrix]))
    print csr_matrix.get_shape(matrix01)


print csr_matrix.get_shape(posts_matrix)
appendMatrix(posts_matrix, regex_matrix)

print("--- %s seconds ---" % (time.time() - start_time))