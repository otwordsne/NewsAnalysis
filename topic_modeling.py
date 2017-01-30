"""
Functions to perform topic modeling on comments,

Running LDA on tokenized comments.
Assigning the LDA topics to a comment
Also, running the LDA to get topics and subtopics

Used in most of my other scripts
"""

import ast
import gensim
from gensim import corpora
import logging
import numpy as np
import pandas as pd
import re
from unidecode import unidecode


# Assign a topic to a comment, given the topics and words with weights
# Topics_word_weights should be a list of dictionaries
# Each dictionary corresponds to one topic. Dictionary keys are words and values are weights
# Returns the topic number. If no topic matches, returns -1
def assign_topic(comment, topics_word_weight):
    topic_num = 0
    temp_list = [0] * len(topics_word_weight)  # hold the weight for each topic
    for topic in topics_word_weight:
        for word in topic:
            if word in comment:
                temp_list[topic_num] += topic[word]
        topic_num += 1  # increment the topic
    max_wieght_topic = max(temp_list)
    if max_wieght_topic == 0:
        return -1
    for i in range(topic_num + 1):
        if temp_list[i] == max_wieght_topic:
            return i


# Run the LDA
# Takes in a list of lists, where each list contains the tokenized words from comments
# also input needs number of topics and words and number of times to go through the corpus
def run_lda(comment_words, n_topics=5, n_words=8, passes=3, iterations=900):
    # logging updates the status so that you know it is running
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dictionary = corpora.Dictionary(comment_words)
    doc_term_matrix = [dictionary.doc2bow(comment) for comment in comment_words]
    ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=n_topics, id2word = dictionary, passes = passes, iterations = iterations)
    topics = ldamodel.print_topics(num_topics=n_topics, num_words=n_words)
    words = get_lda_topic_words(topics)

    topics = overlap_threshold_lda(topics, words, comment_words, n_topics, n_words)
    print topics
    return topics


# Check that there isn't too much overlap in the topics returned from LDA
# If there is, rerun the LDA with one less topic, until no overlap problem
# topics_from_lda is the output from LDA
# words is a list of lists of words, where each list corresponds to one topic
# comment words is a list of list of words corresponding to each comment
def overlap_threshold_lda(topics_from_lda, words, comment_words, num_topics, num_words, threshold=.45):
    topic_overlap = calculate_topic_overlap(words)
    num_topics = len(words)  # topics_from_lda)
    if num_topics == 1:
        return topics_from_lda

    num_words = len(words[0])
    top_words_overlap = overlap_top_topic_words(words)

    while ((max(topic_overlap) > threshold) or top_words_overlap):
        print 'Rerunning LDA because topics had too much overlap'
        num_topics -= 1
        if num_topics == 1:
            return topics_from_lda
        topics_from_lda = run_lda(comment_words, num_topics, num_words)
        words = get_lda_topic_words(topics_from_lda)
        topic_overlap = calculate_topic_overlap(words)
        top_words_overlap = overlap_top_topic_words(words)
    return topics_from_lda


# Get the specific topics for each comment in addition to the category level topics
# Pass in dataframe, list of tokenized comments, number of topics and words
# Passes and iterations are parameters to get convergence
# Passes: number of times LDA sees the corpus (mainly useful with small corpi)
# iterations: limit on how many times LDA will update for each comment
# Returns dataframe with category_level topic
def lda_specific_topics(df, comment_words, n_topics=6, n_words=8, n_passes=2, iterations = 2000):
    topics = run_lda(comment_words, n_topics, n_words, n_passes, iterations = iterations)

    words = get_lda_topic_words(topics)

    topics_word_weight = get_topics(topics)

    # Assign topics to each topic, based on the topic with the greatest weight given the comment
    topic_assignments = [assign_topic(comment, topics_word_weight) for comment in comment_words]

    df['topic'] = topic_assignments
    df['topic_words'] = df['topic'].apply(lambda x: words[x])

    # Groupby topics and then run LDA on each of these subsets
    topic_dfs = df.groupby(['topic'])

    lda_df = pd.DataFrame()
    n_topics = n_topics - 2
    for topic_num, dff in topic_dfs:
        comments = dff['stemmed_comments']
        n_topics = n_topics - 1
        topics = run_lda(comments, n_topics, n_words)
        words = get_lda_topic_words(topics)

        topics_word_weight = get_topics(topics)

        # Assign topics to each topic, based on the topic with the greatest weight given the comment
        topic_assignments = [assign_topic(comment, topics_word_weight) for comment in comments]

        dff['topic_specific'] = topic_assignments
        dff['topic_specific_words'] = dff['topic_specific'].apply(lambda x: words[x])
        lda_df = lda_df.append(dff)
    return lda_df


# Get the associated words in each topic. Returns a list of lists,
# where each list corresponds to one topic
def get_lda_topic_words(topics_from_lda):
    topic_words = [topic[1] for topic in topics_from_lda]

    # Extract the words
    words = [re.sub('\d*', '', x).replace('*','').replace('.','').replace('"','').replace('+','').replace("'",'').split() for x in topic_words]
    return words


# Get the associated weight for each word in each topic. Returns a list of lists,
# where each list corresponds to one topic
def get_lda_topic_word_weights(topics_from_lda):
    topic_words = [topic[1] for topic in topics_from_lda]
    words_probs = [re.findall('\.\d*',x) for x in topic_words]

    #get the probabilities of all the words and chunk them into groups of num_words
    words_probs = [ast.literal_eval(x[i]) for x in words_probs for i in range(len(x))]
    words_probs = [words_probs[i*8:(i+1)*8-1] for i in range(len(topics_from_lda))]
    return words_probs


# Get the words and their weights from each topic from LDA output
# Returns a list of dicionaries, one for each topic
# Dictionary keys are words and values are weights of word in topic
def get_topics(topics_from_lda):
    # Extract the words from LDA output
    words = get_lda_topic_words(topics_from_lda)

    # Extract the word weights from the LDA output
    words_probs = get_lda_topic_word_weights(topics_from_lda)

    topics_word_weight = [dict(zip(words[i], words_probs[i])) for i in range(len(topics_from_lda))]
    return topics_word_weight


# Input: a list of list of words
# Calculate the overlap between the words in each pair of topics
def calculate_topic_overlap(lda_topic_words):
    word_sets = []
    n_words = len(lda_topic_words[0])
    for group in lda_topic_words:
        word_set = set(group)
        word_sets.append(word_set)

    topic_overlap = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            overlap = set.intersection(word_sets[i], word_sets[j])
            percent_overlap = len(overlap) / float(n_words)
            topic_overlap.append(percent_overlap)
    return topic_overlap


# Input: List of lists of words, one for each topic
# Make sure that the top 2 words from each topic are unique
def overlap_top_topic_words(words, top_n_words=2):
    top_topic_words = []
    unique_top_topic_words = set()
    for topic in words:
        for i in range(top_n_words):
            if i >= len(words[0]):
                return len(top_topic_words) != len(unique_top_topic_words)
            top_topic_words.append(topic[i])
            unique_top_topic_words.add(topic[i])
    return len(top_topic_words) != len(unique_top_topic_words)


# Convert a string representation of a list to a list
def convert_string_to_list(str_rep_of_list):
    return [x.replace('[', '').replace(']', '').replace(',', '').replace("'", "").split() for x in str_rep_of_list]


# Strip out the comments with nothing left after preprocessing. df['stemmed_comments']
def strip_uninformative_comments(df):
    #data = pd.read_csv('~/Downloads/kroger_comments_LDAtopics.csv')
    if type(df['stemmed_comments'][0]) is str:
        is_comment = df['stemmed_comments'].apply(lambda x: len(x) > 2)
    else:
        is_comment = df['stemmed_comments'].apply(lambda x: len(x) > 0)
    df = df[is_comment]
    return df


# Classify comments in a dataframe
# Pass in dataframe, the list or series of tokenized comments
# and the words and weights for each topic
# Returns the dataframe with topics and associated words for each comment
def classify_comments(df, comment_words, topics_word_weight):
    topic_assignments = [assign_topic(comment, topics_word_weight) for comment in comment_words]

    df['topic'] = topic_assignments
    df['topic_words'] = df['topic'].apply(lambda x: words[x])
    return df
