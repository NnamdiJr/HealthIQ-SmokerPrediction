__author__ = 'Nnamdi'

import time
start_time = time.time()
import numpy

from sklearn.feature_extraction.text import TfidfTransformer


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


#Creates an array of documents, x, where a document is the array of a user's keywords
user_num = 0
corpus = {}
while user_num <= len(users_vector)-2978:
    corpus[user_num] = posts_matrix[user_num,:]
    user_num += 1


tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)
tfs = tfidf.fit_transform(posts_matrix)

print tfs[10,:]

"""
rs = open('results.txt', 'w')
i = 0
while i <= len(tfs.nonzero()[0])-1:
    a = str(tfs.nonzero()[0][i])
    b = str(feature_names[tfs.nonzero()[1][i]])
    c = str(tfs[tfs.nonzero()[0][i], tfs.nonzero()[1][i]])
    rs.write(a + " - " + b + " - " + c + "\n")
    i += 1
rs.close()
"""

# token_pattern=r'\b[a-zA-Z]{3,}\b' #potential token pattern for tfidf


print("--- %s seconds ---" % (time.time() - start_time))

"""Creates CSV file of user labels associated with each occurrence of a hand picked keyword
lbls = open('keys.csv', 'w')
for keyword in keywords_vector:
    if keyword in lst:
        x = numpy.nonzero(posts_matrix[:,keywords_vector.index(keyword)])
        for index in x[0]:
           output = str(keyword) + " - " + str(index) + " - " + str(labels_vector[index])
           lbls.write(output + "\n")
    continue
lbls.close()
"""