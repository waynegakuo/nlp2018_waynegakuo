
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[2]:


################################################################################################################
# Train classifier while normalizing text
################################################################################################################

def train():
    # setting the data-set into a data-frame for easy management and manipulation using pandas
    doc = pd.read_csv("amazon_cells_labelled.txt", sep='\t', names=['review', 'sentiment'])
    # print(doc.tail())

    # stop words such as 'a', 'is', 'are' are not significant to the corpus for analysis and therefore
    # are stripped from the data set
    wordset = set(stopwords.words('english'))

    # Transforms text to feature vectors that can be used as input to estimator.
    v = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=wordset)

    class_categ = doc.sentiment  # positive and negative classes

    token = v.fit_transform(doc.review)  # tokenizing the reviews provided in the data-set

    # print(class_categ.shape)  # number of observations/reviews
    # print(token.shape)  # number of unique words after tokenizing

    # Splits the class_categ and token arrays into random train and test subsets
    token_train, token_test, class_train, class_test = train_test_split(token, class_categ, random_state=40)

    # training the naive bayes classifier
    naive_train = naive_bayes.MultinomialNB()
    naive_train.fit(token_train, class_train)

    return naive_train, v


# In[3]:


################################################################################################################
# Train classifier without normalizing text
################################################################################################################

def train_u():
    # Not normalizing/tokenizing text
    vu = TfidfVectorizer(use_idf=False, lowercase=False)

    # setting the data-set into a data-frame for easy management and manipulation using pandas
    doc = pd.read_csv("amazon_cells_labelled.txt", sep='\t', names=['review', 'sentiment'])

    class_categ_u = doc.sentiment  # positive and negative classes
    token_u = vu.fit_transform(doc.review)

    # Splits the class_categ and token arrays into random train and test subsets
    token_u_train, token_u_test, class_u_train, class_u_test = train_test_split(token_u, class_categ_u, random_state=40)

    # training the naive bayes classifier
    naive_train_u = naive_bayes.MultinomialNB()
    naive_train_u.fit(token_u_train, class_u_train)

    return naive_train_u, vu


# In[1]:


###################################################################################################################
# Testing with normalized data
###################################################################################################################

def nb(cl, mod, test_file):
    naive_train, v = train()
    file = open(test_file, "r")
    predict_array = []  # initialize array to contain classifier results
    for line in file:
        # treating each line by putting them into an array using an inbuilt panda function
        movie_review_arr = pd.np.array([line])
        movie_vect = v.transform(movie_review_arr)
        class_placed = naive_train.predict(movie_vect)

        # putting the classification results into an array
        predict_array.append(class_placed)

    f = open("nb-n.txt", "w")

    # writing the results into a text file
    for item in predict_array:
        res = str(item)
        f.write(res.strip('[]') + "\n")
    f.close()


# In[2]:


###################################################################################################################
# Testing with not normalized data
###################################################################################################################

def nb_u(cl, mod, test_file):
    naive_train_u, vu = train_u()
    file = open(test_file, "r")
    predict_array_u = []  # initialize array to contain classifier results
    for line in file:
        # treating each line by putting them into an array using an inbuilt panda function
        movie_review_arr = pd.np.array([line])
        movie_vect = vu.transform(movie_review_arr)
        class_placed = naive_train_u.predict(movie_vect)

        # putting the classification results into an array
        predict_array_u.append(class_placed)

    f = open("nb-u.txt", "w")

    # writing the results into a text file
    for item in predict_array_u:
        res = str(item)
        f.write(res.strip('[]') + "\n")
    f.close()


# In[6]:


# Execute 

# accepting arguments from the command line
nb(sys.argv[1], sys.argv[2], sys.argv[3])
nb_u(sys.argv[1], sys.argv[2], sys.argv[3])

if sys.argv[1] == "nb" and sys.argv[2] == "n":
    cl = sys.argv[1]
    mod = sys.argv[2]
    test_file = sys.argv[3]
    print("Naive Bayes Classifier with Normalized Data")
    nb(cl, mod, test_file)

elif sys.argv[1] == "nb" and sys.argv[2] == "u":
    cl = sys.argv[1]
    mod = sys.argv[2]
    test_file = sys.argv[3]
    print("Naive Bayes Classifier Without Normalized Data")
    nb_u(cl, mod, test_file)

