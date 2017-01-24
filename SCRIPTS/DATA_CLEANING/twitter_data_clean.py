"""
Name: Anirudh, Kamalapuram Muralidhar
Masters degree @Indiana University
Objective: To clean the twitter data.
"""

import pandas as pd
from sys import argv
import re
import nltk as nk
from sklearn.feature_extraction import stop_words
from string import punctuation, digits

def read_file_df(input_file):
    """
    Read the input file as a pandas
    dataframe and returns them.
    """

    df = pd.read_csv(input_file)
    # read file as pandas DF
    return df

def remove_names(tweet):
    """
    This function helps to remove any tagged
    names in the tweet and returns them
    """

    return re.sub("@.*?\s", "", tweet)
    # return the tweet with names removed

def remove_stop_words(tweet):
    """
    This function helps to remove stop
    words from the tweet and returns them
    """


    word_list = tweet.lower().split()
    # get all words as a list
    for word in word_list[:]:
        # iterate through each word
        if word in stop_words_list:
            # check if word is a stop word
            word_list.remove(word)
            # remove the stop word
    pos_words_list = nk.pos_tag(word_list)
    # add POS tags
    pos_words_list = [val[0] + '_' + val[1] for val in pos_words_list]
    return " ".join(pos_words_list)

def remove_punctuation_digits(tweet):
    """
    This function helps to remove the
    punctuations and digits from the tweet
    and returns them
    """

    for char in punctuation + digits:
        # iterate through each char
        tweet = tweet.replace(char, "")
        # replace char with empty string
    return tweet

def clean_tweets(df):
    """
    This function helps to clean the tweets.
    """

    for tweet in df["Tweet"]:
        # iterate through each tweet
        actual_tweet = tweet
        # get the actual tweet
        tweet = remove_names(tweet)
        # remove any tagged names in tweet
        tweet = remove_punctuation_digits(tweet)
        # remove the punctuations and digits from the tweet
        tweet = remove_stop_words(tweet)
        # remove stop words from the tweet
        if not tweet:
            tweet = "bad"
        df.Tweet = df.Tweet.replace(actual_tweet, tweet)
        # replace old tweet with cleaned tweet
    return df

def main():
    """
    The main function call.
    """

    df = read_file_df(input_file)
    # get the input file as a pandas DF
    return clean_tweets(df)
    # function call to clean tweets

if __name__ == '__main__':
    # start of the program
    input_file = argv[1]
    # get the input file as a sys argv
    stop_words_list = list(stop_words.ENGLISH_STOP_WORDS)
    # list of stop words
    df = main()
    # function call
    df.to_csv(input_file + '.csv')
    # write out clean data to a CSV file
