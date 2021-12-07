import nltk
nltk.download('stopwords')

import myapp

from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from numpy import linalg as la
import json
import re
import unidecode

#Data with original text of the tweet
with open('dataset_tweets_WHO.txt') as f:
    json_data1 = json.load(f)

#Data with the preprocessed text of a tweet
with open('dataset_tweets_WHO.txt') as f:
    json_data = json.load(f)

#Text prepocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
def cleanText(text):
    text = unidecode.unidecode(text)
    a = text.lower() #put everything in lowercase
    result = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",a) #cleaning data
    result = result.split() #tokenize
    result = [word for word in result if word not in stop_words ] #removing stop words
    result = [stemmer.stem(word) for word in result] #stemming (looking for the root)
    return result

# Putting the preprocessed text back into the dictionary
for i in json_data.keys():
    json_data[i]['full_text'] = cleanText(json_data[i]['full_text'])

#Creating the list of preprocessed tweets
all_tweets = []
for i in json_data.keys():
     all_tweets.append(json_data[i]['full_text'])
       
