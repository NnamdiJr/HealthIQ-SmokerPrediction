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
topic_matrix = np.empty([1, 100], dtype=int)


def fit_lda(X, vocab, num_topics=100, passes=50):
    """Fit LDA from a scipy CSR matrix (X)."""
    print('fitting lda...')
    return LdaModel(matutils.Sparse2Corpus(X, documents_columns=False), num_topics=num_topics, passes=passes,
                    chunksize=10, update_every=1, id2word=dict([(i, s) for i, s in enumerate(vocab)]))


def print_topics(lda, vocab, n=10):
    """Print the top words for each topic."""
    topics = lda.show_topics(num_topics=100, num_words=n, formatted=False)
    for ti, topic in enumerate(topics):
        print('topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic)))


def user_doc_bow(user):
    """Transform user's document vector into list of tuples"""
    user_doc = []
    user_array = np.zeros(100, dtype=int)
    i = 0
    while i < len(keywords_vector)-1:
        user_doc.append((i, user[0, i]))
        i += 1
    topics = lda[user_doc]
    topic_idx = [item[0] for item in topics]
    np.put(user_array, topic_idx, 1)
    return user_array


lda = fit_lda(posts_matrix, keywords_vector)
print("--- %s seconds ---" % (time.time() - start_time))
print_topics(lda, keywords_vector)
print("--- %s seconds ---" % (time.time() - start_time))

for row_index in range(posts_matrix.shape[0])[:5]:
    user_idx = posts_matrix[row_index, :]
    topic_matrix = np.vstack((topic_matrix, user_doc_bow(user_idx)))

    print("Running Time: %s seconds ||| Current Row:" % (time.time() - start_time)), row_index

np.savetxt('LDA_matrix.txt', topic_matrix[1:, :]) #save topic matrix to a text file.