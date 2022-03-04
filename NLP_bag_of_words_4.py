# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:26:00 2022

@author: Home
"""

import nltk

paragraph = """ APJ Abdul kalam came into this world on 15th October 1931, at a time when India was under British occupation. He was born to a Tamil Muslim family in Tamil Nadu. His father was a boat owner while his mother was a housewife. Furthermore, Kalam had five siblings and was the youngest of the lot.

In school, Kalam was an average student but was still hardworking and bright. I think this certainly is a great motivation for all the average students out there. Being average, you must never ever underestimate yourself and continue doing the hard work.

An Illustrious Scientist
In the year 1960, APJ Abdul Kalam’s graduation took place from Madras Institute of Technology. The association of Kalam took place with the Defence Research & Development Service (DRDS). Furthermore, he joined as a scientist at the Aeronautical Development Establishment of the Defence Research and Development Organisation. These were the beginning achievements of his prestigious career as a scientist.

Big achievement for Kalam came when he was the project director at ISRO of India‘s first-ever Satellite Launch Vehicle (SLV- III). This satellite was responsible for the deployment of the Rohini satellite in 1980. Moreover, Kalam was highly influential in the development of Polar Satellite Launch Vehicle (PSLV) and SLV projects.

Both projects were successful. Bringing enhancement in the reputation of Kalam. Furthermore, the development of ballistic missiles was possible because of the efforts of this man. Most noteworthy, Kalam earned the esteemed title of “The missile Man of India”.

The Government of India became aware of the brilliance of this man and made him the Chief Executive of the Integrated Guided Missiles Development Program (IGMDP). Furthermore, this program was responsible for the research and development of Missiles. The achievements of this distinguished man didn’t stop there.

More success was to come in the form of Agni and Prithvi missiles. Once again, Kalam was influential in the developments of these missiles. It was during his tenure in IGMDP that Kalam played an instrumental role in the developments of missiles like Agni and Prithvi. Moreover, Kamal was a key figure in the Pokhran II nuclear test.
"""
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

sentences = nltk.sent_tokenize(paragraph)
corpus =[]

for i in range(len(sentences)):
     review = re.sub('[^a-zA-Z]', ' ', sentences[i])
     review = review.lower()
     review = review.split()
     review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
     review = ' '.join(review)
     corpus.append(review)
     
     
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 

bow = cv.fit_transform(corpus).toarray()     



     
    

