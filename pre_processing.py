# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:52:35 2019

@author: prettyyang2
"""

'''
convert the original data to a format that Bert could understand

'''
import codecs
import os
import pandas as pd
import csv
from sklearn.preprocessing import MultiLabelBinarizer 



#from logging import getLogger

def custom_character_handler(exception):
    '''
    using the space to replace the unencoded byte
    '''
    return (" ", exception.end)
    #no_punct.translate(str.maketrans"\n\t\r","  ")

if __name__ == "__main__":

    codecs.register_error("custom_character_handler", custom_character_handler)
    
    path = './multi-labelled data/'
    multi_hot_labels = []
    ds = pd.DataFrame() 
    
    for root, dirs, files in os.walk(path):        
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path,encoding='utf8',errors='custom_character_handler') as f:
                lines=f.read().strip().split('\n')
                l = lines[0].strip()
                l = l.strip('\ufeff\n\r ')
                c = lines[1:]
                
                entry = pd.DataFrame(data=[[l, ''.join(c)]], columns = ['label', 'text'])
                ds = ds.append(entry)
                multi_hot_labels.append(l.split(", "))
    
    #convert to multi-hot encoding           
    mlb = MultiLabelBinarizer()                                                          
    labels = mlb.fit_transform(multi_hot_labels)  
    label_list = list(mlb.classes_)
    
    #save tsv
    ds.reset_index(drop=True).set_index('label').to_csv('./data/data.tsv',sep='\t',header=False)

    #save the label_list
    with open('./data/label_list.tsv',mode='w',newline='') as file_handle:
        tsv_output = csv.writer(file_handle,delimiter='\t')
        tsv_output.writerow(label_list)
    
    
    

    


   
        
    

        

        
   
                
                
                
    
    
    

