# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:52:45 2019

@author: prettyyang2
"""
import matplotlib.pyplot as plt
import csv
import re
import numpy as np
from collections import Counter

def count_words(text):
    '''count how many words in the text'''
    words=0
    for lines in text:         
        lines = lines.lower()
        lines = re.sub('\W', ' ', lines) 
        lines = re.sub('\s+', ' ', lines) 
        lines = lines.strip()        
        data = lines.split(' ')
        words+=len(data)
    return words

if __name__ == "__main__":
    with open('label_list.tsv',encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        label_list = []
        for line in reader:
            label_list.extend(line)
            
    with open('data.tsv.',encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        labels = []
        dic = {}
        words=[]
        for line in reader:
            temp = line[0].split(', ')
            word = count_words(line[1:])
            words.append(word)
            n = len(temp)
            if n not in dic:
                dic[n] = 1
            else:
                dic[n] += 1
            labels.extend(temp)
                     
#count num of classes
    classes_count = Counter(labels)
    classes_count = sorted(classes_count.items(),key = lambda item:item[1])    
    classes=[x[0] for x in classes_count] 
    num=[x[1] for x in classes_count] 
    
    plt.barh(np.arange(len(classes)),num)
    for xx, yy in zip(np.arange(len(classes)),num):
        plt.text(yy+0.6, xx-0.2, str(yy), ha='center')
    plt.yticks(np.arange(len(classes)),classes)
    plt.xlabel("number")

    
    plt.title("number of classes")
    plt.tight_layout()
    plt.savefig('num_classes.png',transparent=True)
    plt.show()
        
    #count How many samples have multiple labels?
    x = list(dic.keys())
    y = list(dic.values())
    
    plt.bar(x, y, width=0.3)
    for xx, yy in zip(x,y):
        plt.text(xx, yy, str(yy), ha='center')
    my_x_ticks = np.arange(1, 2.1, 1)
    my_y_ticks = np.arange(0, 166, 20)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel("number of labels")
    plt.title("How many samples have multiple labels?")
    plt.savefig('labels.png', transparent=True)
    plt.show()
    
    
    
    plt.hist(words, bins=30)
    my_y_ticks = np.arange(0, 21, 1)
    plt.yticks(my_y_ticks)
    # x label
    plt.xlabel("number of words")
    # y label
    plt.ylabel("number of articles")
    # title
    plt.title("The distribution of the number of words in articles")
    plt.savefig('hist.png', transparent=True)
    plt.show()
