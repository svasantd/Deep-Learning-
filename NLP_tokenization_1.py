

import nltk
nltk.download()

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
#tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)

#tokenizing words
words = nltk.word_tokenize(paragraph)