__author__ = 'Nnamdi'

import nltk
from nltk.stem.porter import PorterStemmer

"""Opens the pickled file"""
f=open('C:\Users\Nnamdi\Desktop\HealthIQ\Smoker Analysis\smoking_analytic_data_1.pkl', 'rb')


"""Import pickled file data into numpy array called data"""
data = numpy.load(f)
data = numpy.array(data)
f.close()


"""Rename data elements by their corresponding properties"""
likes_matrix = data[0]
posts_matrix = data[1]
keywords_vector = data[2] #size is 587002
users_vector = data[3] #size is 2985
labels_vector = data[4]

"""Function that returns a single string of all the keywords from a user's posts"""
def userBagOfWords(matrix, user):
    key_num = 0
    string = ""
    while key_num <= len(keywords_vector)-1:
        if matrix[user, key_num] > 0:
            string += ((keywords_vector[key_num] + " ") * matrix[user, key_num])
            key_num += 1
        else:
            key_num += 1
            continue
    return string


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

"""Creates an array of documents, x, where a document is the bag of words of a user's keywords"""
user_num = 0
corpus = {}
while user_num <= len(users_vector)-2978:
    corpus[user_num] = userBagOfWords(posts_matrix, user_num)
    user_num += 1