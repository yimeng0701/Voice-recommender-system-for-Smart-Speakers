# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:35:44 2019

@author: prettyyang2
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize

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

def evaluate_score(Y_test,predict): 
    """
    Y_test: array
    predict: array
    """
    temp = Y_test ^ predict
    each_acc = 1-np.sum(temp,axis=0)/len(Y_test)
    loss = hamming_loss(Y_test,predict)

    accuracy = accuracy_score(Y_test,predict)

    return each_acc, loss*100, accuracy*100

def clean_data(lines):
    '''only keep characters and numbers'''
    ans=[]
    for data in lines: 
        data = data.lower()
        data = re.sub(r'\W', ' ', data)
        data = re.sub(r'\s+', ' ', data)
        data = re.sub(r'\d', ' ', data) #remove numbers
        data = data.strip(' ')
        ans.append(data)
    return ans


def encoding(context, methods='TF-IDF'):
    if methods == 'TF-IDF':
        vectorizer = TfidfVectorizer(stop_words = 'english', max_features=1000)
        ds_features= vectorizer.fit_transform(context)        
        voc = vectorizer.get_feature_names()
    elif methods =='BOW':
        vectorizer = CountVectorizer(stop_words = 'english')
        ds_features = vectorizer.fit_transform(context).toarray()
        voc = vectorizer.get_feature_names()
    return ds_features,voc

    
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
    
    ds_features,voc = encoding(context,methods="TF-IDF")

    #K-fold cross-validation:
    acc_knn = []
    loss_knn = []
    best_acc_knn = 0
    
    acc_dt = []
    loss_dt = []
    best_acc_dt = 0
    kf = KFold(n_splits=5,random_state=None,shuffle=True)
    i = 0
    for train_index, test_index in kf.split(ds_features):
        i+=1
        X_train, X_test = ds_features[train_index], ds_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # knn classifier
        clf1 =  KNeighborsClassifier(n_neighbors=3)
        clf1.fit(X_train, y_train) 
        r1 = clf1.predict (X_test)
        # DT classifier
        clf2 = tree.DecisionTreeClassifier()
        clf2.fit(X_train, y_train) 
        r2 = clf2.predict (X_test)
        r2 = r2.astype(np.int32)

    
        _,loss1, acc1 = evaluate_score(y_test,r1)
        acc_knn.append(acc1)
        loss_knn.append(loss1)
        #for i in range(len(label_list)):
            #print("knn, label:{}, acc:{}".format(label_list[i],each_acc[i]))
        
        _,loss2, acc2 = evaluate_score(y_test,r2)
        acc_dt.append(acc2)
        loss_dt.append(loss2)
        #for i in range(len(label_list)):
            #print("DT, label:{}, acc:{}".format(label_list[i],each_acc2[i]))
        # save the best model
        if acc1 > best_acc_knn:
            best_acc_knn = acc1
            joblib.dump(clf1, "./models/knn_model.pkl")
        
        if acc2 > best_acc_dt:
            best_acc_dt = acc2
            joblib.dump(clf2, "./models/dt_model.pkl")
        
            
    avg_acc_knn = np.mean(acc_knn)
    avg_loss_knn = np.mean(loss_knn)
    avg_acc_dt = np.mean(acc_dt)
    avg_loss_dt = np.mean(loss_dt)
    print ("knn accuracy: {}".format(avg_acc_knn))
    print ("knn loss: {}".format(avg_loss_knn))
    print ("dt accuracy: {}".format(avg_acc_dt))
    print ("dt loss: {}".format(avg_loss_dt))
      
    
    
   
    
   