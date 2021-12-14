import random

#Imports text_procesing
import nltk
nltk.download('stopwords')
import collections
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from numpy import linalg as la
import json
import re
import unidecode

from myapp.search.objects import ResultItem, Document

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


def cleanCorpus(corpus):
    print("> Start preprocessing the corpus")
    # Putting the preprocessed text back into the dictionary
    for i in corpus.keys():
        corpus[i]['full_text'] = cleanText(corpus[i]['full_text'])
    print("> Cleaning completed")

    #Creating the list of preprocessed tweets
    clean_terms = []
    for i in corpus.keys():
        clean_terms.append(corpus[i]['full_text'])
    print("> Cleaned Corpus loaded")
    return corpus, clean_terms


def build_demo_results(corpus: dict, search_id):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    size = len(corpus)
    ll = list(corpus.values())
    for index in range(random.randint(0, 40)):
        item: Document = ll[random.randint(0, size)]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

    # for index, item in enumerate(corpus['Id']):
    #     # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
    #     res.append(DocumentInfo(item.Id, item.Tweet, item.Tweet, item.Date,
    #                             "doc_details?id={}&search_id={}&param2=2".format(item.Id, search_id), random.random()))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        results = build_demo_results(corpus, search_id)  # replace with call to search algorithm

        # results = search_in_corpus(search_query)
        ##### your code here #####

        return results
