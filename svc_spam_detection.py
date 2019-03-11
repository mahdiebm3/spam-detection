# -*- coding: utf-8 -*-


################ importing libraries

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline



############## Define patterns for text cleaning, these patterns includes: html tags, websites, stopwords
#### also stemming and lemmatizibg words using nltk package

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
combined_pat = r'|'.join((pat_1, pat_2))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'
negations_ = {"isn't":"is not", "can't":"can not","couldn't":"could not", "hasn't":"has not",
            "hadn't":"had not","won't":"will not",
            "wouldn't":"would not","aren't":"are not",
            "haven't":"have not", "doesn't":"does not","didn't":"did not",
             "don't":"do not","shouldn't":"should not","wasn't":"was not", "weren't":"were not",
            "mightn't":"might not",
            "mustn't":"must not"}

negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')


STOPWORDS = stopwords.words('english')

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
 
def clean_text(text):
    stripped = re.sub(combined_pat, '', text)
    stripped = re.sub(www_pat, '', stripped)
    cleantags = re.sub(html_tag, '', stripped)
    lower_case = cleantags.lower()
    neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    tokenized_text = word_tokenize(letters_only.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS]
    alpha_numeric = [t for t in cleaned_text if t.isalpha()]
    stemmed = [stemmer.stem(w) for w in alpha_numeric]
    lem = [lemma.lemmatize(k, pos = 'v') for k in stemmed]
    cleaned = ' '.join(w for w in lem)
    return cleaned


#### loading news and spams dataset

news = pd.read_csv('./data/news.csv')
news.drop(['Unnamed: 0'], axis = 1, inplace = True)
news['topic'] = 'real'
news.rename({'article': 'text'}, axis = 1, inplace = True)

spams = pd.read_csv('./data/spams.csv')
spams.drop(['Unnamed: 0', 'non_alphabetic_ratio'], axis = 1, inplace = True)


data = pd.concat([spams, news], axis = 0, ignore_index = True)

data['topic'].value_counts()


################ Data Preprocessing: 1) apply text cleaning function to text column and 2) convert categorical variable into numbers

data['cleaned_text'] = data.text.apply(clean_text)

data['labels'] = data.topic.map({'real':0, 'general': 1, 'marketing': 2, 'algorithmic': 3})


##### splitting data into train and test set
x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], data['labels'], test_size = 0.2, random_state = 10)


############### fit frequency based word embeddings into our data set to turn text into wordvectors

vectorizer = TfidfVectorizer(lowercase = True, stop_words = STOPWORDS)
vectorizer.fit(x_train)
x_train_vect = vectorizer.transform(x_train)
x_test_vect = vectorizer.transform(x_test)

############# Build our classifier with Linear Support vector machine

model = SVC(C = 1, kernel = 'linear', class_weight = 'balanced')
model.fit(x_train_vect, y_train)

y_pred = model.predict(x_test_vect)

cm = confusion_matrix(y_test, y_pred)   ########## confusion matrix for test set
                                               
pipeline = make_pipeline(vectorizer, model) #### save our model with pipeline function for future analysis
                                              
                                                                                       
def predict(text):
    
    score = pipeline.predict([clean_text(text)])
    
    if score == 0:
        topic = 'real news'
    elif score == 1:
        topic = 'Genral spam'
    elif score == 2:
        topic = 'Marketing spam'
    elif score == 3:
        topic = 'Algorithmic spam'
        
    return topic
    
    
predict(text = "Market Update: Bitcoin - $3,776.80 Bitcoin Cash - $125.09 Ethereum - $128.58 Litecoin - $46.74 XRP - $0.31 #Cryptos")    
    
from sklearn.externals import joblib  
  
joblib.dump(pipeline, './pipeline.pickle')    
    
    
     
    
    
    
    
    











