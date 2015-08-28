__author__ = 'Nnamdi'

import time
start_time = time.time()
import numpy as np
from scipy.sparse import csr_matrix
from gensim import matutils
from gensim.models.ldamodel import LdaModel

#Loading pickle file data into numpy array called data
f = open('smoking_1_analytic_data_mapreduce.pkl', 'rb')
data = np.load(f)
data = np.array(data)
f.close()

#Loading smoking posts_matrix data
posts_matrix = data[0] #11616 rows x 605107 columns
rows = csr_matrix.get_shape(posts_matrix)[0]
users_vector = data[3]
labels_vector = data[4]
keywords_vector = data[2]

#Empty matrix for load columns
topic_matrix = np.empty([1, 100])


def fit_lda(X, vocab):
    """Fit LDA from a scipy CSR matrix (X)."""
    print('fitting lda...')
    return LdaModel(matutils.Sparse2Corpus(X, documents_columns=False), num_topics=100, passes=1, iterations=500,
                    chunksize=1000, update_every=1, id2word=dict([(i, s) for i, s in enumerate(vocab)]))


def print_topics(lda):
    """Print the top words for each topic."""
    topics = lda.show_topics(num_topics=100, num_words=10, formatted=False)
    for ti, topic in enumerate(topics):
        print('topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic)))


def user_doc_bow(user):
    """Transform user's document vector into list of tuples"""
    user_doc = []
    user_array = np.zeros(100)
    nonzero_idx = [item for item in user.nonzero()[1]]
    for i in nonzero_idx:
        user_doc.append((i, user[0, i]))
    topics = lda[user_doc]
    topic_dict = {topic[0]: topic[1] for topic in topics}
    np.put(user_array, topic_dict.keys(), topic_dict.values())
    return user_array

#Fitting LDA model on posts_matrix
lda = fit_lda(posts_matrix, keywords_vector)
print("--- %s seconds ---" % (time.time() - start_time))
print_topics(lda)
print("--- %s seconds ---" % (time.time() - start_time))


#Creates topic matrix row by row.
for row_index in range(posts_matrix.shape[0]):
    user_idx = posts_matrix[row_index, :]
    topic_matrix = np.vstack((topic_matrix, user_doc_bow(user_idx)))

    print("Running Time: %s seconds ||| Current Row:" % (time.time() - start_time)), row_index

np.savetxt('LDA_matrix01.txt', topic_matrix[1:, :]) #save topic matrix to a text file.