"""
Name: Anirudh, Kamalapuram Muralidhar
Indiana University
Objective: This program main aim is to
preprocess the data before training them on a classifier.
"""

import numpy as np
import ast
from sys import argv
from string import punctuation,digits
from HTMLParser import HTMLParser
from nltk import pos_tag
from sklearn.feature_extraction import stop_words

def get_data(line):
    """
    This function helps to extract the required columns
    alone to perform data cleaning and returns them.
    The headers that are extracted are
    1. asin
    2. reviewerID
    3. reviewText
    4. rating
    """

    review_dict = ast.literal_eval(line)
    # convert the data line to a python dictionary
    return (review_dict['asin'], review_dict['reviewerID'],
           review_dict['reviewText'], review_dict['overall'])

def parse_html_entity(review):
    """
    This function helps to parse the
    HTML entities and return them
    """

    review = h.unescape(review)
    # unescape the HTML entities
    return review.encode('utf8')

def to_lower_case(review):
    """
    This function helps to convert the
    data to lower case and return them
    """

    return review.lower()

def remove_stop_words(review_list, stop_words_arr):
    """
    This function helps to remove the
    stop words in the dataset and returns them.
    Arguments:
        1. review_list: The review to be processed as a list.
        2. stop_words_arr: numpy array of stop words
    """

    for word in review_list[:]:
        # iterate through each word in review
        if word in stop_words_arr:
            # check if the word is "stop word"
            review_list.remove(word)
            # remove the word from the list
    return " ".join(review_list)

def remove_punctuation(review):
    """
    This function helps to remove the
    punctuations in the given string and returns
    them.
    """

    for punc in punctuation+digits:
        # iterate over each punctuation
        review = review.replace(punc, "")
        # replace each punctuation to an empty string
    return review

def add_pos_tags(review):
    """
    Add parts of speech tags to the words
    and return them.
    """

    pos_list = pos_tag(review.split())
    # get the pos tagger list
    return " ".join([word[0] + '_' + word[1] for word in pos_list])


def data_cleaning(review, stop_words_arr):
    """
    This function helps to clean the review data and
    return them. Various functions are called for this
    procedure.

    Arguments:
        1. review: The review data to processed as a string.
        2. stop_words_arr: A numpy array of stop words.
    """

    review = parse_html_entity(review)
    # function call to parse HTML entity
    review = to_lower_case(review)
    # function call to convert to lower case
    review = remove_punctuation(review)
    # function call to remove punctuations
    review_list = review.split()
    # get the reviews words as a numpy array
    review = remove_stop_words(review_list, stop_words_arr)
    # function call to remove stop words
    review = add_pos_tags(review)
    # add parts of speech tags to the words
    return review

def read_file(input_file):
    """
    Read the input file and iterate through
    each line and process the data.
    """

    count = 0
    file_obj = open(input_file)
    # open the file object for operations
    stop_words_arr = list(stop_words.ENGLISH_STOP_WORDS)
    # load the stop words as a numpy array
    for line in file_obj:
        # iterate through each line in the file
        if count == 20000:
            break
        asin, review_id, review, rating = get_data(line)
        # get the required columns alone
        print review_id
        review = data_cleaning(review, stop_words_arr)
        # function call to clean data
        if review and rating:
            f.write('{},{},{},{}\n'.format(asin,review_id,review,rating))
            # write data to the file


if __name__ == '__main__':
    # start of the program
    input_file = argv[1]
    # get the input file as a sys argv
    h = HTMLParser()
    # initialize the HTML parser
    output_file = input_file + '.csv'
    f = open(output_file, 'w')
    # open file in append mode
    a = read_file(input_file)
    # function call
    f.close()
    # close the file
