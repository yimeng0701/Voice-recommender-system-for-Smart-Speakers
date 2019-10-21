# AskVOCO 
![home_page](https://github.com/yimeng0701/Voice-recommender-system-for-Smart-Speakers/blob/master/verySimpleApp/home_page.png "Logo Title Text 1")
AskVOCO is a online multi-label news classification platform. It is also a part of the project which is about building better news recommender system for smart speakers, collaborating with the company VOCO. The long-term plan of this project is to do news classification, which is aleady done, build user profile and finally make custormized news recommendation.

## Getting Started

AsoVOCO makes classification based on Bert. knn and decision tree are also tried as baseline. To get everything works, please follow those instructions:

### Prerequisites

The project is based on Python 3.6. Please make sure that all those packaged are installed.

```
bert-tensorflow          1.0.1
Flask                    1.1.1
gensim                   3.8.0
matplotlib               3.0.2
nltk                     3.4.5
numpy                    1.15.3
pandas                   0.23.4
scikit-learn             0.20.0
scikit-multilearn        0.2.0
tensorflow               1.12.0
```

### Download Bert
To use the pre-trained Bert Model, you need to download:
1. The source code of Bert from Google-research: download [here](https://github.com/google-research/bert.git).
2. The pre-trained model: considering the computational cost, I use [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip).

## Processing Pipeline

### Data
I got the data from the company VOCO, which contains around 200 manually labelled articles. 1000 other articles from different news websites are also crawled. There are 11 classess in the data: Football, Tech, Baseball, Basketball, Canadian National, Hockey, International, Finance, US National, Science, Soccer. Each article belongs to one or more classes. Following is the basic analysis of the data:
![# of classes](https://github.com/yimeng0701/Voice-recommender-system-for-Smart-Speakers/blob/master/data/EDA/num_classes.png "Logo Title Text 1")
![# of words](https://github.com/yimeng0701/Voice-recommender-system-for-Smart-Speakers/blob/master/data/EDA/hist.png "Logo Title Text 1")

### Preprocessing
Preprocessing includes converting all letters to the lower case, removing stop words, removing punctuations, remocing stop words and tokenization. These could be done by running:
1. pre-processing.py
2. large-processing.py
Also the above scripts could convert the data to a form that Bert could use.
### Feature Extraction
Converting words to vectors that classifiers can recognize is a key procedure for classification. For simple classifiers, TF-IDF is used for feature extraction. For Bert, the input is just the one-hot representing of the article since it could do embedding inside the model.

### Models
To implement the simple classifiers(knn and decision tree): run simple_classier.py.

To fine-tune Bert: run mybert.py. Most of this part of code is from https://github.com/google-research/bert/blob/master/run_classifier.py

Notice: Since it's a multi-label classification task, the softmax layer should be changed to sigmoid layer. Besides, the loss function used here is tf.nn.sigmoid_cross_entropy_with_logits.

### Metrics
1. Hamming Loss:  the fraction of labels that are incorrectly predicted
2. Subset Accuracy: the percentage of samples that have all their labels classified correctly

### Results
| Classifier        | Subset Accuracy(%)           | Hamming Loss  |
| ------------- |:-------------:| -----:|
| pre-trained Bert| 81.57 | 2.06 |
| knn      | 64.14      |  4.59 |
| Decision Tree | 55.22      |    6.68 |


## AskVOCO web App

The web app was deployed based on Flask and AWS EC2. All needed files are contained in /verySimpleApp. 

Enter the news in the textbox and then click "Classify":
![Page_2](https://github.com/yimeng0701/Voice-recommender-system-for-Smart-Speakers/blob/master/verySimpleApp/Page_2.png "Logo Title Text 1")
The website will return you the classification result:
![Page_3](https://github.com/yimeng0701/Voice-recommender-system-for-Smart-Speakers/blob/master/verySimpleApp/Page_3.png "Logo Title Text 1")

## Background
### Bert
Bert (Bidirectional Encoder Representations from Transformers) is a latest language representation model released by Google in Oct 2018. It is pre-trained on Wikipedia and could be fine-tuned on your own data. Following is the overall pre-training and fine-tuning procedures for BERT:
![Bert](https://github.com/yimeng0701/Voice-recommender-system-for-Smart-Speakers/blob/master/models/bert.png "Logo Title Text 1")

Both the pre-trained part and the fine-tuning part use the same architectures except for the output layers. The weights from the pre-trained model are used to initialize the fine-tuning model and the output layers vary for different NLP tasks.

## Authors

* **Yimeng Wu** - *2019C. TO AI Fellow of Insight Data Science* - [Yimeng Wu](https://github.com/yimeng0701/)

## Acknowledgments
* https://github.com/google-research/bert/blob/master/run_classifier.py

## Reference
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
* Zhang, M. L., & Zhou, Z. H. (2007). ML-KNN: A lazy learning approach to multi-label learning. Pattern recognition, 40(7), 2038-2048.
