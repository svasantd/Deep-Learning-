# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:56:27 2022

@author: Home
"""
#importing dataset
import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                       names=['label','message'])

# data cleaning and processing
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

ps = PorterStemmer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    corpus.append(review)
    
    
#creating bag of words that is creating vectors and feature of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()  

y = pd.get_dummies(messages['label'])  
y = y.iloc[:,1].values


#train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#training the model 

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()

spam_detect_model.fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
