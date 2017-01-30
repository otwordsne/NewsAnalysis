"""
Functions for preprocessing text data.
e.g. Tokenizing, cleaning, getting word counts.

Used in most of my other scripts
"""

import ast
from autocorrect import spell
from collections import Counter
from functools32 import lru_cache
import gensim
from gensim import corpora
import glob
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pandas as pd
import re
import sys
import time
from unidecode import unidecode


# Removes the stopwords and invalid words.
# Then tokenizes text, corrects spelling, and tags part of speech
# Input, string of text, a tokenizer, and a set of stopwords
# Returns list of tokens with corresponding tag
def clean_and_tag(text, tokenizer, stop_words):
    try:
        text = text.decode('utf-8').encode('ascii', 'ignore').lower()  # apply(lambda x: x.lower())
    except Exception:
        pass
    try:
        text = text.encode('ascii', 'ignore').lower()  # apply(lambda x: x.lower())
    except Exception:
        pass
    try:
        text = text.decode('latin-1').encode('ascii', 'ignore').lower()
    except Exception:
        return [('', '')]

    # Spell check, and filter for valid words
    tokens = [spell(clean_repeated_chars(word)) for word
              in tokenizer.tokenize(text) if word not in stop_words and len(word) < 50]
    cleaned_tokens = [word.lower() for word in tokens if check_vowels(word) and len(word) < 25]
    tagged_tokens = nltk.pos_tag(cleaned_tokens)
    return tagged_tokens


# Remove characters that are repeated more than twice
def clean_repeated_chars(word):
    repeats = re.compile(r'(.)\1\1+')  # matches characters repeated >2 times
    return re.sub(repeats, r'\1\1', word)


# Check that a word contains at least one vowel
def check_vowels(word):
    word = word.lower()
    vowels = 'aeiouy'
    for char in word:
        if char in vowels:
            return True
    return False


# Tokenize the text and remove stopwords and non-alphabetical words
def tokenize_text(text):
    stop = set(stopwords.words('english'))
    text = text.lower()
    tokenizer = RegexpTokenizer(r'[a-z]+')  # strips non-alphabetical characters
    cleaned_tokens = [spell(word) for word in tokenizer.tokenize(text) if word not in stop]
    # tagged_tokens = nltk.pos_tag(cleaned_tokens)
    return cleaned_tokens


# *** This is the function that takes the longest in the preprocessing  ***
# Input: series of lists of tokens and a lemmatizer.
# Returns series of lists with lemmatized tokens
def lemmatize_tokens(tokens_col, lemmatize_func):
    # lemma = WordNetLemmatizer()
    # noun_tokens = tokens_col.apply(lambda row: get_nouns(row))
    tokens_col = tokens_col.apply(lambda row: [get_wordnet_pos(word) for word in row if get_wordnet_pos(word)[1] != ''])
    lemmatized_words = tokens_col.apply(lambda row: [lemmatize_func(word[0], word[1]) for word in row])  # [lemma.lemmatize(x) for x in noun_tokens]
    return lemmatized_words


# Input a series of lists of tokens and and stemmer.
# Returns stemmed tokens
def stem_tokens(tokens_col, stem_func):
    print 'stem'
    # stemmer = PorterStemmer()
    stems = tokens_col.apply(lambda row: [stem_func(spell(word)) for word[0] in row if word is not None])
    return stems


# Cleans and tags the tokens, then stems them
def clean_text(text_col, tokenizer, lemma):
    # print 'clean_text'
    lemma = WordNetLemmatizer()
    # cleaned_article = remove_links(article)
    tagged_tokens = text_col.apply(lambda row: clean_and_tag(x, tokenizer))

    cleaned_tokens = lemmatize_tokens(tagged_tokens, lemma)
    return cleaned_tokens


# Convert to wordnet part of speech from UPenn treebank
def get_wordnet_pos(word_tag_tuple):
    # print 'get_wordnet'
    tag = word_tag_tuple[1]
    if tag.startswith('J'):  # adjectives
        tag = 'a'
    elif tag.startswith('V'):  # verbs
        tag = 'v'
    elif tag.startswith('N'):  # nouns
        tag = 'n'
    elif tag.startswith('R'):  # adverbs
        tag = 'r'
    else:                      # default tag in nltk lemmatizer is noun
        tag = ''
    return (word_tag_tuple[0], tag)


# Input: series of comments
# Output: series of tokenized and stemmed comments
def get_tokens(comment_col):
    tokenizer = RegexpTokenizer(r'[a-z]+')
    stop = set(stopwords.words('english'))
    tokenized_comments = comment_col.apply(lambda x: clean_and_tag(x, tokenizer, stop))
    wnl = WordNetLemmatizer()
    lemma_func = lru_cache(maxsize=60000)(wnl.lemmatize)
    stemmed_comments = lemmatize_tokens(tokenized_comments, lemma_func)
    return stemmed_comments


# Convert comments to dataframe of words and counts
def get_word_counts(df, text_col='Comment', threshold=8):
    tokenizer = RegexpTokenizer(r'[a-z]+')
    stop = set(stopwords.words('english'))

    # version of lemmatizer that caches previously computed results
    wnl = WordNetLemmatizer()
    lemma_func = lru_cache(maxsize=60000)(wnl.lemmatize)

    comments = df[text_col]
    tokenized_comments = comments.apply(lambda x: clean_and_tag(x, tokenizer, stop))
    stemmed_comments = lemmatize_tokens(tokenized_comments, lemma_func)

    word_count_df = count_freq(stemmed_comments)
    word_count_df.columns = ['word_count']
    word_count_df['len_ge'] = word_count_df['word_count'].apply(lambda x: x > threshold)
    word_count_df = word_count_df[word_count_df['len_ge'] == 1]
    word_count_df = word_count_df.sort_values('word_count', ascending=False)
    word_count_df['num_comments'] = len(comments)

    return word_count_df

# Take in a dataframe of comments formatted in the ground truth file format with a column of labelled topics
def get_word_feature_data(dff, wordcount_dff=None, text_col='Comment', relevant_cols=['UID','Comment','stemmed_comments','topic'], threshold=8):
    if wordcount_dff is not None:
        print 'wordcount yes'
        word_count_df = wordcount_dff
    else:
        print 'wordcount no'
        word_count_df = get_word_counts(dff, text_col, threshold)

    word_features = list(word_count_df.index)

    features = pd.DataFrame(dff[relevant_cols])
    for word in word_features:
        features[word] = dff['stemmed_comments'].apply(lambda x: x.count(word))
    return features


# Takes a column or series of tokens
# returns the number of occurrences of each word
def count_freq(tokens_col):
    print 'count frequency'
    words = [str(word.encode('ascii','ignore')) for sublist in list(tokens_col) for word in sublist]
    count_freq = dict(Counter(words))
    df = pd.DataFrame.from_dict(count_freq, orient='index')
    return df


# Get the sentiment scores for comments
# Input: dataframe with a comments column
def analyze_sentiment(df):
    comment_col = df.columns[[x.startswith(('open_end', 'Comment','comment')) for x in df.columns]][0]

    sid = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df[comment_col].apply(lambda x:
                              [sid.polarity_scores(y) for y
                              in tokenize.sent_tokenize(x.encode('ascii', 'ignore'))])

    # # If don't filter out, next block may get errors
    # df['temp'] = df['sentiment_scores'].apply(lambda x: len(x) > 0)
    # df = df[df['temp']]
    # del df['temp']

    # returns score of each sentence in each comment, list of scores for multi-sentence comments
    df['pos_sentiment'] = df['sentiment_scores'].apply(lambda x: np.mean([sentence['pos'] for sentence in x if len(x)>0]))
    df['neg_sentiment'] = df['sentiment_scores'].apply(lambda x: np.mean([sentence['neg'] for sentence in x if len(x)>0]))
    df['overall_sentiment'] = df['sentiment_scores'].apply(lambda x: np.mean([sentence['compound'] for sentence in x if len(x)>0]))
    return df


# Convert a string representation of a list to a list
def convert_string_to_list(str_rep_of_list):
    return [x.replace('[', '').replace(']', '').replace(',', '').replace("'",'').split() for x in str_rep_of_list]


# Load a pickle file
def load_file(file_name):
    sm_data_path = '/Users/rchen/Downloads/data/'
    print sm_data_path + file_name
    data_file = max(glob.iglob(sm_data_path + file_name), key=os.path.getctime)

    df = pd.read_pickle(data_file)
    open_end_col = [x for x in df.columns if x.startswith(('open_end', 'Comment', 'comment'))][0]
    df = df[df[open_end_col] != '']

    try:
        df['response_date'] = pd.to_datetime(df['response_date'])
    except Exception:
        pass

    return df
