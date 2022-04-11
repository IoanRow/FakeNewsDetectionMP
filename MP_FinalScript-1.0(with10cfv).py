"""
@author: ior2
Version: 0.1
"""
import pandas as pd

#preparing text for classification
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

from time import time  
import numpy as np
import os

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc
import seaborn as sn

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_table('C:/CS - Degree/Year3/S2/Major project/Datasets/train.csv')

def preprocessing(dataset):
    #Convert text to lowercase
    dataset.text = dataset['text'].str.lower()
    
    def remove_punc(text):
        all_words = [char for char in text if char not in string.punctuation]
        punctuation_removed = ''.join(all_words)
        return punctuation_removed
    dataset.text = dataset['text'].apply(remove_punc)
    
    #Removal stopwords    
    #stopwords = [ 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'I', 'that', 'had', 'on', 'for', 'were', 'was']
    stopwords = []
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    print("data has been preprocessed")
    return dataset

#dataset balencing
def undersampling():
    #data frame of individual labels.
    #fake
    fake_data = []
    count = 0
    for d in dataset.label:
        if(d == 1):
            fake_data.append(dataset.text[count])
        count +=1
    
    print("only fakenews:")
    only_fake_data =pd.DataFrame(fake_data, columns = ['text'])
    
    only_fake_data['label'] = 1
    print(only_fake_data)
    
    #true
    true_data = []
    count = 0
    for d in dataset.label:
        if(d == 0):
            true_data.append(dataset.text[count])
        count +=1
        
    print("only truenews:")
    only_true_data =pd.DataFrame(true_data, columns = ['text'])    
    only_true_data['label'] = 0
    
    #cutting the data
    final_fake_data = only_fake_data.sample(1500)
    final_true_data = only_true_data.sample(1500)
    
    man_bal_data = pd.concat([final_fake_data, final_true_data], axis=0)
    
    y = man_bal_data['label']
    x_train, x_test, y_train, y_test = train_test_split(man_bal_data['text'], man_bal_data.label, test_size=0.2, random_state=42, stratify=y)

    fig, ax = plt.subplots()
    fig.suptitle("Dataset Balance", fontsize = 12)
    dataset['label'].reset_index().groupby('label').count().sort_values(by="index").plot(kind='barh', legend=False, ax=ax).grid(axis='x')
    plt.show()
    
    print("data has been undersampled")
    return x_train, x_test, y_train, y_test
def oversampling():
    
    #data frame of individual labels.
    #fake
    fake_data = []
    count = 0
    for d in dataset.label:
        if(d == 1):
            #print(d, "\n")
            #print(dataset.text[count])
            fake_data.append(dataset.text[count])
        count +=1
    
    print("only fakenews:")
    only_fake_data =pd.DataFrame(fake_data, columns = ['text'])
    #print(only_fake_data.head())
    
    only_fake_data['label'] = 1
    print(only_fake_data)
    
    
    #true
    true_data = []
    count = 0
    for d in dataset.label:
        if(d == 0):
            #print(d, "\n")
            #print(dataset.text[count])
            true_data.append(dataset.text[count])
        count +=1
        
    print("only truenews:")
    only_true_data =pd.DataFrame(true_data, columns = ['text'])
    #print(only_true_data.head())
    
    only_true_data['label'] = 0
    print(only_true_data)
    
    
    #cutting the data
    print(only_fake_data.shape)
    print(only_true_data.shape)
    
    final_fake_data = only_fake_data.sample(2296, replace = True)    
    final_true_data = only_true_data
    
    man_bal_data = pd.concat([final_fake_data, final_true_data], axis=0)
    
    y = man_bal_data['label']
    x_train, x_test, y_train, y_test = train_test_split(man_bal_data['text'], man_bal_data.label, test_size=0.2, random_state=42, stratify=y)
    
    print(y_train.value_counts())
    print(x_train)
    print(y_train)

    fig, ax = plt.subplots()
    fig.suptitle("Dataset Balance", fontsize = 12)
    man_bal_data['label'].reset_index().groupby('label').count().sort_values(by="index").plot(kind='barh', legend=False, ax=ax).grid(axis='x')
    plt.show()
    print("data has been undersampled")
    return x_train, x_test, y_train, y_test

#tfidf
def apply_tfidf():
    #Convert a collection of raw documents to a matrix of TF-IDF features.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    #fit the 
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    print("TF-IDF used")
    return tfidf_train, tfidf_test

#word embeddings
def preloading_wordembeddings():
    # Global parameters
    #root folder
    root_folder='.'
    data_folder_name='data'
    glove_filename='glove.6B.50d.txt'
    
    train_filename='train.csv'
    # Variable for data directory
    DATA_PATH = os.path.abspath(os.path.join(root_folder, data_folder_name))
    glove_path = os.path.abspath(os.path.join(DATA_PATH, glove_filename))
    
    # Both train and test set are in the root data directory
    train_path = DATA_PATH
    test_path = DATA_PATH
    
    #Relevant columns
    TEXT_COLUMN = 'text'
    TARGET_COLUMN = 'target'
    
    # We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format 
    # that will be loaded in the next section
    from gensim.scripts.glove2word2vec import glove2word2vec
    
    #glove_input_file = glove_filename
    word2vec_output_file = glove_filename+'.word2vec'
    glove2word2vec(glove_path, word2vec_output_file)
    
    
    
    from gensim.models import KeyedVectors
    # load the Stanford GloVe model
    word2vec_output_file = glove_filename+'.word2vec'
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return model
def apply_wordembddings(x_train, x_test, model):
    
    X_train = []
    for d in x_train:
        X_train.append(d)
    
    X_test = []
    for da in x_test:
        X_test.append(da)
        
    
        # Set a word vectorizer
    vectorizer = Word2VecVectorizer(model)
    # Get the sentence embeddings for the train dataset
    x_train = vectorizer.fit_transform(X_train)
    # Get the sentence embeddings for the test dataset
    x_test = vectorizer.transform(X_test)
    
    print("word embeddings used")
    return x_train, x_test

#dataset balencing
def apply_smote(x_train, y_train):
    sm = SMOTE(random_state=42)
    x, y = sm.fit_resample(x_train, y_train)
    print("SMOTE applied")
    return x, y 
   
#classification 
def classification(x_train, y_train, x_test, y_test, classifier):
    # create the model, train it, print scores#
    #clf = RandomForestClassifier()
    #clf = MLPClassifier()
    clf = classifier
    
    clf.fit(x_train, y_train)
    
    
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    print("train score:", train_score)
    print("test score:", test_score)
    
    y_pred = clf.predict(x_test)

    print(metrics.classification_report(y_test, y_pred,  digits=5))
    plot_confussion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)
    
    return train_score, test_score
    
class Word2VecVectorizer:
  def __init__(self, model):
    print("Loading in word vectors...")
    self.word_vectors = model
    print("Finished loading in word vectors")

  def fit(self, data):
    pass

  def transform(self, data):
    # determine the dimensionality of vectors
    v = self.word_vectors.get_vector('king')
    self.D = v.shape[0]

    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split()
      vecs = []
      m = 0
      for word in tokens:
        try:
          # throws KeyError if word not found
          vec = self.word_vectors.get_vector(word)
          vecs.append(vec)
          m += 1
        except KeyError:
          pass
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X


  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)

# Create the confussion matrix
def plot_confussion_matrix(y_test, y_pred):
    ''' Plot the confussion matrix for the target labels and predictions '''
    cm = confusion_matrix(y_test, y_pred)

    # Create a dataframe with the confussion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                  range(cm.shape[1]))
    #plt.figure(figsize = (10,7))
    # Plot the confussion matrix
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, annot=True,fmt='.0f',annot_kws={"size": 10})# font size
    plt.show()
def plot_roc_curve(y_test, y_pred):
    ''' Plot the ROC curve for the target labels and predictions'''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc= auc(fpr,tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


#nessesary for word embeddign
#model = preloading_wordembeddings()

preprocessing(dataset)

#different methods of splitting data and balencing - smote used later 
#x_train, x_test, y_train, y_test = undersampling()
#x_train, x_test, y_train, y_test = oversampling()

#overall time
overall_time = time()

#kfold cross validation
from sklearn.model_selection import StratifiedKFold
splits = 2
folds = StratifiedKFold(n_splits=splits)

total_train_score = 0.000000000000000
total_test_score  = 0.000000000000000

for y_train, y_test in folds.split(dataset.text, dataset.label):
    
    #time per fold
    fold_time = time()
    
    x_train, x_test, y_train, y_test = dataset.text[y_train], dataset.text[y_test], dataset.label[y_train], dataset.label[y_test]
    
    #Word representations
    x_train, x_test = apply_tfidf()
    #x_train , x_test = apply_wordembddings(x_train, x_test, model)
    
    
    #Dataset balencing - using SMOTE
    x_train , y_train = apply_smote(x_train , y_train)
    
    #classifiers
    #classifier = RandomForestClassifier()
    #classifier = MLPClassifier()
    classifier = KNeighborsClassifier()
    
    train_score, test_score = classification(x_train, y_train, x_test, y_test, classifier)
    
     
    total_train_score = total_train_score + train_score
    total_test_score  = total_test_score  + test_score

    print("Time taken: ", fold_time)
    
    print()
    print("-----------------------------------------")
    print()

avg_train = total_train_score/splits 
avg_test = total_test_score/splits

print("average train score: ", avg_train)
print("average test score: ", avg_test)

print("The overall time of classification is:", overall_time)

clf = classifier
# Predicting the Test set results
y_pred = clf.predict(x_test)

print(metrics.classification_report(y_test, y_pred,  digits=5))
plot_confussion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred)