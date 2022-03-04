# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:34:30 2022

@author: Home
"""

import nltk
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re


paragraph = """ APJ Abdul kalam came into this world on 15th October 1931, at a time when India was under British occupation. He was born to a Tamil Muslim family in Tamil Nadu. His father was a boat owner while his mother was a housewife. Furthermore, Kalam had five siblings and was the youngest of the lot.

In school, Kalam was an average student but was still hardworking and bright. I think this certainly is a great motivation for all the average students out there. Being average, you must never ever underestimate yourself and continue doing the hard work.

An Illustrious Scientist
In the year 1960, APJ Abdul Kalam’s graduation took place from Madras Institute of Technology. The association of Kalam took place with the Defence Research & Development Service (DRDS). Furthermore, he joined as a scientist at the Aeronautical Development Establishment of the Defence Research and Development Organisation. These were the beginning achievements of his prestigious career as a scientist.

Big achievement for Kalam came when he was the project director at ISRO of India‘s first-ever Satellite Launch Vehicle (SLV- III). This satellite was responsible for the deployment of the Rohini satellite in 1980. Moreover, Kalam was highly influential in the development of Polar Satellite Launch Vehicle (PSLV) and SLV projects.

Both projects were successful. Bringing enhancement in the reputation of Kalam. Furthermore, the development of ballistic missiles was possible because of the efforts of this man. Most noteworthy, Kalam earned the esteemed title of “The missile Man of India”.

The Government of India became aware of the brilliance of this man and made him the Chief Executive of the Integrated Guided Missiles Development Program (IGMDP). Furthermore, this program was responsible for the research and development of Missiles. The achievements of this distinguished man didn’t stop there.

More success was to come in the form of Agni and Prithvi missiles. Once again, Kalam was influential in the developments of these missiles. It was during his tenure in IGMDP that Kalam played an instrumental role in the developments of missiles like Agni and Prithvi. Moreover, Kamal was a key figure in the Pokhran II nuclear test.
"""

#preprocessing the data
text = re.sub('[^a-zA-Z]', ' ', paragraph)
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ', text)
text = re.sub(r'\s+',' ', text)

# preparing the dataset

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in set(stopwords.words('english'))]
    
    
model = Word2Vec(sentences, min_count=1)

words = list(model.wv.index_to_key)

words[1]

vector = model.wv[1]

similar = model.wv.most_similar(words[1])  