__author__ = 'Nnamdi'
import os, codecs
import numpy
from scipy.sparse import vstack, hstack, csr_matrix


#Loading pickle file data into numpy array called data
f = open('C:\Users\Nnamdi\Desktop\HealthIQ\Smoker Analysis\smoking_1_analytic_data_mapreduce.pkl', 'rb')
data = numpy.load(f)
data = numpy.array(data)
f.close()


#Loading smoking posts_matrix data
posts_matrix = data[1] #11616 rows x 605107 columns
users_vector = data[3]


with open('C:\Users\Nnamdi\Desktop\HealthIQ\Smoker Analysis\collocation_smoker_regexs.txt','r') as f:
    queries = [l.strip() for l in f]

hold_matrix = csr_matrix((1, csr_matrix.get_shape(posts_matrix)[1])) #Empty matrix with 1 row


for query in queries:
    os.system("grep -i %s smoking_1_noUser0.txt > tmp.txt" % query)
    results = codecs.open('tmp.txt', encoding='utf-8')
    userIDs = [line[10:16].strip() for line in results]
    dic = {}
    for user in set(userIDs):
        dic[user] = userIDs.count(user)

    for user in users_vector:
        if user in dic.keys():
            hold_matrix = vstack([hold_matrix, matrix02[numpy.where(regex_userIDs==user)[0][0],:]])
            continue
        hold_matrix = vstack([hold_matrix, numpy.zeros(csr_matrix.get_shape(matrix02)[1])])



