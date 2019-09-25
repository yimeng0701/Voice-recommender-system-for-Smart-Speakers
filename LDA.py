from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import word_tokenize

 # Create a corpus from a list of texts
#common_dictionary = Dictionary(common_texts)
#common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
 # Train the model on the corpus.
#lda = LdaModel(common_corpus , id2word=common_dictionary,
#                               alpha='auto',
#                               num_topics=10,
#                               passes=5)

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:35:44 2019

@author: prettyyang2
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

import csv
import os
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import re

from sklearn.preprocessing import MultiLabelBinarizer 

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score


def clean_data(lines):
    '''remove punctuation'''
    ans=[]
    for data in lines:        
        data = data.lower()
        data = re.sub('\W', ' ', data)
        data = re.sub('\s+', ' ', data)
        data = data.strip(' ')
        ans.append(data)
    return ans

    
if __name__ == "__main__":

    path = './data/'
    
    multi_hot_labels = []
    context=[]
    
    with open(os.path.join(path,'data.tsv'),encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            labels = line[0].split(', ')
            multi_hot_labels.append(labels)   
            c = line[1:]
            c = clean_data(c)
            context.extend(c)
    #convert to multi-hot encoding 
    mlb = MultiLabelBinarizer()                                                          
    labels = mlb.fit_transform(multi_hot_labels)  
    label_list = list(mlb.classes_)
    
    token_context = [word_tokenize(x) for x in context]
    token_list = []
    for x in token_context:
        temp = [i for i in x if not i in stop_words]
        token_list.append(temp)
    token_context = [clean_data(x) for x in token_list]
    del token_list
    common_dictionary = Dictionary(token_context)
    common_corpus = [common_dictionary.doc2bow(text) for text in token_context]
    # Train the model on the corpus.
    lda = LdaModel(common_corpus , id2word=common_dictionary,
                               alpha='auto',
                               num_topics=3,
                               passes=5)
    print(lda.show_topic(2,20))
  
    