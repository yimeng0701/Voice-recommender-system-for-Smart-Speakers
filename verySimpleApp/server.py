# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import csv

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

clf_knn = joblib.load('../models/knn_model.pkl')
voc = joblib.load('../data/voc.p')
label_list = _read_tsv('../data/label_list.tsv')[0]

def clean_data(data):
    '''only keep characters and numbers'''    
    data = data.lower()
    data = re.sub(r'\W', ' ', data)
    data = re.sub(r'\s+', ' ', data)
    data = re.sub(r'\d', ' ', data) #remove numbers
    data = data.strip(' ')
    return data

def encoding(context, voc, methods='TF-IDF'):
    if methods == 'TF-IDF':
        vectorizer = TfidfVectorizer(stop_words = 'english', vocabulary = voc)
        ds_features= vectorizer.fit_transform(context)        
        voc = vectorizer.vocabulary_
    elif methods =='BOW':
        vectorizer = CountVectorizer(stop_words = 'english',vocabulary=voc)
        ds_features = vectorizer.fit_transform(context).toarray()
        voc = vectorizer.vocabulary_
    return ds_features

def id2label(prediction,label_list):
    labels=[]
    for i in range(len(label_list)):
        if prediction[i]==1:
            labels.append(label_list[i])
    return labels

# Create the application object
app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output',methods=["GET"])
def tag_output():  
       
    # Pull input
    raw_input =request.args.get('input_text')

                    
    some_input = [clean_data(raw_input)]
    some_input = encoding(some_input,voc)
    # Case if empty
    if raw_input == '':
      return render_template("insight.html", my_input = raw_input,my_form_result="Empty")
                              
    else:
      raw_labels = clf_knn.predict(some_input)
      some_output = id2label(raw_labels[0],label_list)
      n = len(some_output)
      return render_template("insight.html",
                              my_input=raw_input,
                              my_output=some_output,
                              length = n,
                              my_form_result="NotEmpty")

@app.route('/back',methods=["GET","POST"])
def go_back():
    return render_template('index.html')  # render a template


if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

