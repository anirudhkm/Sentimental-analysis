"""
Name: Anirudh, Kamalapuram Muralidhar
Indiana University
Objective: To develop classifier for
text classification.
"""

import numpy as np
import pandas as pd
from sys import argv
from sklearn.preprocessing import scale
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def buildWordVector(text, size, w2v):
    """
    This function helps to build the word vector model
    and returns the vector.
    Arguments:
        1. text: The input text as a string.
        2. The size of the data.

    Return values:
        1. Returns the vector model.
    """

    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def get_word_2_vec_obj(reviews, n_dim = 150, min_count = 10):
    """
    This function helps to create the word2vec model and
    returns them
    Arguments:
        1. reviews: The input reviews as an numpy array.
        2. n_dim: The no of dimensions with default value to 100.
        3. min_count: The min count to build the word2vec model.

    Return values:
        1. The word2vec model.
        2. The no of dimensions.
    """

    w2v = Word2Vec(size = n_dim, min_count = min_count)
    # develop the word2vec model
    w2v.build_vocab(reviews)
    # build vocabulary based on the reviews
    w2v.train(reviews)
    # train the data
    return w2v, n_dim

def train_vectors(reviews, n_dim, w2v):
    """
    This function helps to develop the train vector models
    and returns them.
    Arguments:
        1. reviews: The reviews data as a numpy array.
        2. n_dim: The no of dimensions.
        3. w2v: The word2vec model.
    Return values:
        1. The train_vec models.
    """

    train_vecs = np.concatenate([buildWordVector(z, n_dim, w2v) for z in reviews])
    # build the train vectors
    train_vecs = scale(train_vecs)
    # scale the data
    return train_vecs



def get_data(input_file):
    """
    This function helps to read the input file and
    reads them and returns the reviews and ratings.

    Arguments:
        1. input_file: The input file as a string.

    Return values:
        1. The reviews as a numpy array.
        2. The ratings as a numpy array.
    """

    data = pd.read_csv(input_file)
    # read the input file as a pandas dataframe.
    try:
        reviews = data['Review']
    except:
        reviews = data['Tweet']

    ratings = data['Rating']
    return reviews, ratings, data

def plot_roc_curve(clf, y_test, test_vecs, model_name):
    """
    This function helps to plot the ROC curve
    and saves the file to the desktop.
    """

    predicted_prob = clf.predict_proba(test_vecs)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, predicted_prob)
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,label= '{0}: Area = {1:.2f}'.format(model_name, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate", fontsize = 18)
    plt.ylabel("True Positive Rate", fontsize = 18)
    plt.title("ROC curve for amazon review data with no POS (10 fold CV)", fontsize = 18)
    plt.legend(loc = 'lower right')
    plt.show()


def report_generation(actual, predicted):
    """
    This function helps to generate the report
    for classification
    """

    accuracy = np.mean(actual == predicted)
    # accuracy of the classifier
    report = metrics.classification_report(actual, predicted)
    # generate report
    matrix = metrics.confusion_matrix(actual, predicted)
    # generate confusion matrix
    print "Accuracy score is {}".format(accuracy)
    print "Classification report\n".format(report)
    print "Confusion Matrix\n {}".format(matrix)

def data_analysis(reviews, ratings):
    """
    The main function call.
    """

    w2v, n_dims = get_word_2_vec_obj(reviews)
    # build and train the word2vec model
    train_vecs = train_vectors(reviews, n_dims, w2v)
    # get the reviews as a vectors
    models_dict = {}
    # empty dict
    svm_classifier = svm.SVC(probability = True)
    models_dict['SVM'] = svm_classifier
    # SVM classifier
    logistic = LogisticRegression()
    models_dict['Logistic'] = logistic
    # Logistic regression classifier
    naive_bayes = GaussianNB()
    models_dict['Naive Bayes'] = naive_bayes
    # Naive Bayes classifier
    fit_model = {}
    # empty dict
    for model in models_dict:
        print "Algorithm used " + model
        clf = models_dict[model].fit(train_vecs, ratings)
        # build the classifier model
        fit_model[model] = clf
        # add fitted model to the dict
        predicted = cross_val_predict(clf, train_vecs, ratings, cv = 10, n_jobs = -1)
        report_generation(ratings, predicted)
        # function call to print report
        plot_roc_curve(clf, ratings, train_vecs, model)
    return fit_model

def out_domain_test(amazon_models, twitter_data):
    """
    This function tests the models developed with amazon
    on twitter data.
    """


    ratings = twitter_data['Rating']
    try:
        reviews = twitter_data['Review']
    except:
        ratings = twitter_data['Tweet']

    w2v, n_dims = get_word_2_vec_obj(reviews)
    # build and train the word2vec model
    test_vecs = train_vectors(reviews, n_dims, w2v)
    # get the reviews as a vectors
    print amazon_models
    for model in amazon_models:
        # iterate through each model
        predicted = amazon_models[model].predict(test_vecs)
        report_generation(ratings, predicted)
        # function call to print report
        plot_roc_curve(amazon_models[model], ratings, test_vecs, model)
        # plot ROC curve

if __name__ == '__main__':
    # start of the program
    input_file_amazon = argv[1]
    # get the input file as a sys argv
    twitter_file = argv[2]
    # twitter file as sys argv
    amazon_reviews, amazon_ratings, amazon_df = get_data(input_file_amazon)
    # get the reviews and ratings from the data file
    tweet_reviews, tweet_ratings, tweet_df = get_data(twitter_file)
    ## get the tweets and ratings from the data file
    tweet_df = tweet_df.iloc[np.random.permutation(len(tweet_df))]
    amazon_df = amazon_df.iloc[np.random.permutation(len(amazon_df))]
    # shuffle the data
    combined = pd.concat([amazon_df.iloc[:1500], tweet_df.iloc[:9000]])
    ## combine amazon and tweet validation
    new_rating = combined['Rating']
    try:
        new_review = combined['Review']
    except:
        new_rating = combined['Tweet']
    amazon_models = data_analysis(amazon_reviews, amazon_ratings)
    # function call for amazon data analysis
    gen_models = data_analysis(new_review, new_rating)
    out_domain_test(gen_models, amazon_df)
    out_domain_test(gen_models, tweet_df)
    # function call
