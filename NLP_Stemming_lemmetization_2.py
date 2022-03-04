# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:19:45 2022

@author: Sanket Dahule
"""
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """
Widely hailed as a masterpiece of rhetoric, King's speech invokes pivotal documents in American history, 
including the Declaration of Independence, the Emancipation Proclamation, 
and the United States Constitution. Early in his speech, 
King alludes to Abraham Lincoln's Gettysburg Address by saying 
"Five score years ago ..." In reference to the abolition of slavery 
articulated in the Emancipation Proclamation, King says: 
"It came as a joyous daybreak to end the long night of their 
captivity." Anaphora (i.e., the repetition of a phrase at the beginning of sentences) 
is employed throughout the speech. Early in his speech, King urges his audience to seize the moment; 
"Now is the time" is repeated three times in the sixth paragraph. The most widely cited example of anaphora is found in the often quoted phrase "I have a dream", which is repeated eight times as King paints a picture of an integrated and unified America for his audience. Other occasions include "One hundred years later", "We can never be satisfied", "With this faith", "Let freedom ring", and "free at last". King was the sixteenth out of eighteen people to speak that day, according to the official program

"""
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

#stemming

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
    
sentences    
