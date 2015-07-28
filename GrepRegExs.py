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


for query in queries:
    os.system("grep -i %s smoking_1_noUser0.txt > tmp.txt" % query)
    results = codecs.open('tmp.txt', encoding='utf-8')
    userIDs = [line.split()[0][10:] for line in results]
    dic = {}
    temp_array = numpy.zeros((csr_matrix.get_shape(posts_matrix)[0],)) #Zeros array

    for user in set(userIDs):
        dic[user] = userIDs.count(user)

    for user in dic.keys():
        if user in users_vector:
            numpy.put(temp_array, [users_vector.index(user)], dic[user])
        continue

    temp_array = csr_matrix(temp_array)
    posts_matrix = csr_matrix(vstack([posts_matrix, temp_array]))



