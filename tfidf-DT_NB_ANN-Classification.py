import pandas as pd

import numpy as np

import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


dataset = pd.read_table('C:/CS - Degree/Year3/S2/Major project/Datasets/train.csv')

def preprocessing(dataset):
    
    #Convert text to lowercase
    dataset.text = dataset['text'].str.lower()
    #print(dataset.text, "\n")

    #Removal of punctuation 
    def remove_punc(text):
        all_words = [char for char in text if char not in string.punctuation]
        punctuation_removed = ''.join(all_words)
        return punctuation_removed

    dataset.text = dataset['text'].apply(remove_punc)
    
    #print(dataset.text, "\n")
    


    #Removal of stopwords 
    
    #possible list, however this lowers classification acuracy
    #stopwords = [ 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'I', 'that', 'had', 'on', 'for', 'were', 'was']
    stopwords = []
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    print("Removed stop words")
    print(dataset.text)
    print()
    
    return dataset


import nltk
from nltk import tokenize

def term_frequency():

    token_space = tokenize.WhitespaceTokenizer()

    def counter(label, column_text, quantity):
        all_words = ' '.join([label for label in label[column_text]])
        token_phrase = token_space.tokenize(all_words) 
        frequency = nltk.FreqDist(token_phrase)
        data_frequency = pd.DataFrame({"word": list(frequency.keys()),"Frequency": list(frequency.values())})
    
        print(frequency)
        print(data_frequency)
    
    counter(dataset[dataset["label"] == 0], "text", 2296)
    counter(dataset[dataset["label"] == 1], "text", 1547)


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier

def classify(x_train, x_test, y_train, y_test ):
    
    #Decision Tree classifier
    pipe = Pipeline([('vect', CountVectorizer()),('termfreqidv', TfidfTransformer()),
                     ('model', DecisionTreeClassifier(criterion='entropy', max_depth=20, splitter='best', random_state=42))])
    model = pipe.fit(x_train, y_train)
    
    prediction = model.predict(x_test)
    print("DT - Overall Classification Acuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
    
    #Naive Bayes
    pipe = Pipeline([('vect', CountVectorizer()),('termfreqidv', TfidfTransformer()),('model', BernoulliNB())])
    model = pipe.fit(x_train, y_train)
        
    prediction = model.predict(x_test)
    print("BNB - Overall Classification Acuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    #ANN
    pipe = Pipeline([('vect', CountVectorizer()),('termfreqidv', TfidfTransformer()),
                     ('model', MLPClassifier(hidden_layer_sizes=10))])
    model = pipe.fit(x_train, y_train)
        
    prediction = model.predict(x_test)
    print("ANN - Overall Classification Acuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


preprocessing(dataset)
term_frequency()

#split data
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset.label, test_size=0.2, random_state=42)

#DT classifier
classify(x_train, x_test, y_train, y_test)



