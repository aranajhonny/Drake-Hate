#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Partially adapted from flebel on GitHub at http://bit.ly/1ThAsJL.

import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from user_blacklist import user_blacklist
from word_blacklist import word_blacklist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

with open('sentim_analyzer.pk1', 'rb') as f:
    sentim_analyzer = pickle.load(f)

with open('classifier.pk1', 'rb') as f:
    classifier = pickle.load(f)

with open('trainer.pk1', 'rb') as f:
    trainer = pickle.load(f)

TWITTER_SEARCH_LIMIT = 350


# Load blacklists from file
# TODO: This is clunky af! Can I make it more Pythonic? Maybe move to
# helper fn?

# Store the ID of the last tweet we retweeted in a file
# so we don't retweet things twice!

# Put all of the tweets into a list so we can filter them
with open('test_tweets.txt', 'r', encoding='utf-8') as f:
    tweets = f.readlines()

# TODO: Handle emojis better! Right now tweet.text.split() is
# tokenizing tweets only at whitespace. It'd be nice to recognize
# a string of emojis and process them all individually, rather
# than as a collective 'word'.

# Filter tweets using blacklist
tweets = [tweet for tweet in tweets if not any(
    word.lower() in word_blacklist for word in tweet.split())]
'''tweets = [tweet for tweet in tweets
          if tweet.author.screen_name not in user_blacklist]
'''
# TODO: Handle emojis better! Right now tweet.text.split() is
# tokenizing tweets only at whitespace. It'd be nice to recognize
# a string of emojis and process them all individually, rather
# than as a collective 'word'.

# Filter tweets using blacklist

for tweet in tweets:
    analysis = sentim_analyzer.classify(word_tokenize(tweet))
    if analysis == 'negative':
        print(analysis)
        print(sid.polarity_scores(tweet))
        print(tweet)
        print()
# Write last retweeted tweet id to file
# with open(last_id_file, 'w') as file:
#     file.write(str(last_tweet_id))
