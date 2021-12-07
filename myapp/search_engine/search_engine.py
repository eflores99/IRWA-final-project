import random

from myapp.core.utils import get_random_date, load_documents_corpus
#import myapp.search_engine.text_processing
#import myapp.search_engine.index_ranking

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

# imports indes_ranking

import math

# -------------------------------TEXT PROCESSING--------------------------------------------------


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

#------------------------------------------------------INDEX RANKING--------------------------------------------------------------
def create_index_tfidf(terms, corpus):
    """
    Implement the inverted index and compute tf, df and idf
    
    Argument:
    corpus -- collection of Wikipedia articles
    terms -- clean corpus (terms)
    
    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(list)
    tf = defaultdict(list)  #term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  #document frequencies of terms in the corpus
    idf = defaultdict(float)

    for i in range(len(corpus)):  # Remember, corpus contain all tweets
        page_id = corpus[str(i)]['id']
        terms = corpus[str(i)]['full_text']

        ## ===============================================================        
        ## create the index for the **current page** and store it in current_page_index
        ## current_page_index ==> { ‘term1’: [current_doc, [list of positions]], ...,‘term_n’: [current_doc, [list of positions]]}

        ## Example: if the curr_doc has id 1 and his text is 
        ##"web retrieval information retrieval":

        ## current_page_index ==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}

        ## the term ‘web’ appears in document 1 in positions 0, 
        ## the term ‘retrieval’ appears in document 1 in positions 1 and 4
        ## ===============================================================

        current_page_index = {}

        for position, term in enumerate(terms):  ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[page_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document. 
            # posting ==> [current_doc, [list of positions]] 
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term]+=1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = np.round(np.log(float(len(corpus)/ df[term])), 4)

    return index, tf, df, idf

def rank_documents(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title
    
    Returns:
    Print the list of ranked documents
    """

    # We are interested only on the element of the docVector corresponding to the query terms 
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # We call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    #HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26            
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  # TODO: check if multiply for idf

    # Calculate the score of each doc 
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot
    
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)
    #print ('\n'.join(result_docs), '\n')
    return result_docs

def search_tf_idf(query, index, top, idf, tf, corpus):
    query = cleanText(query)
    docs = set([posting[0] for posting in index[query[0]]])
    print("Starting terms intersection")
    for term in query[1:]:
        try:
            # store in term_docs the ids of the docs that contain "term"                        
            term_docs=[posting[0] for posting in index[term]]
            
            # docs = docs Union term_docs
            docs = docs.intersection(set(term_docs))
        except:
            #term is not in index
            pass
    docs = list(docs)
    print("Finished tems intersection")
    ranked_docs = rank_documents(query, docs, index, idf, tf)
    print("Finished ranking docs")
    print(ranked_docs[0])
    return retrieve_docs(ranked_docs, top, corpus)


def retrieve_docs(ranked_docs, top, corpus):
    
    """
    Retrieve the documents in the required format
    
    Argument:
    docs -- collection of tweets
    top -- the number of tweets to retrieve
    
    Returns:
    doc_info - the collection of top tweets retrieved in the required format
    """
    print("Starting retrieve_docs")
    doc_info = []
    for d_id in ranked_docs[:top]:
        for j in corpus.keys():
            if(d_id == corpus[j]['id']):
                if 'media' in corpus[j]['entities'].keys():
                    title= corpus[j]["full_text"][:20]
                    description = corpus[j]["full_text"]
                    doc_date = corpus[j]["created_at"]
                    url = corpus[j]["entities"]["media"][0]['url']
                    doc_info.append(DocumentInfo(title, description, doc_date, url))
                else:
                    title= corpus[j]["full_text"][:20]
                    description = corpus[j]["full_text"]
                    doc_date = corpus[j]["created_at"]
                    doc_info.append(DocumentInfo(title, description, doc_date, ''))       
    print("Finished retrieve_docs")
    return doc_info

#-------------------------------------------SEARCH ENGINE--------------------------------------------
def build_demo_data():
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    samples = ["Messier 81", "StarBurst", "Black Eye", "Cosmos Redshift", "Sombrero", "Hoags Object",
            "Andromeda", "Pinwheel", "Cartwheel",
            "Mayall's Object", "Milky Way", "IC 1101", "Messier 87", "Ring Nebular", "Centarus A", "Whirlpool",
            "Canis Major Overdensity", "Virgo Stellar Stream"]

    res = []
    for index, item in enumerate(samples):
        res.append(DocumentInfo(item, (item + " ") * 5, get_random_date(),
                                "doc_details?id={}&param1=1&param2=2".format(index), random.random()))
    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    print(res)
    return res


class SearchEngine:
    """educational search engine"""
    #i = 12345
    index = []
    tf = []
    df = []
    idf = []

    def create_index(self, clean_terms, corpus):
        self.index, self.tf, self.df, self.idf = create_index_tfidf(clean_terms, corpus)

    def search(self, search_query, corpus):
        print("Search query:", search_query)
        top = 10
        results = search_tf_idf(search_query, self.index, top, self.idf, self.tf, corpus)
        
        #results= []
        ##### your code here #####
        #results = build_demo_data()  # replace with call to search algorithm
        ##### your code here #####

        return results


class DocumentInfo:
    def __init__(self, title, description, doc_date, url):#, ranking):
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.url = url
        #self.ranking = ranking

       
