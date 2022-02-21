import nltk
#nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()



def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stemming(words):
    return stemmer.stem(words.lower())
    

def bagOfWords(tokenizedSentence,allWords):
    tokenizedSentence=[stemming(i) for i in tokenizedSentence]
    bag=np.zeros(len(allWords),dtype=np.float32)
    for i,word in enumerate(allWords):
        if word in tokenizedSentence:
            bag[i]=1.0
    
    return bag

""" l="Bye! Come back again soon."
print(l)
l=tokenize(l)
print(l) """

""" words=["Organize","organizes","organizing"]
stemmedWords=[stemming(w) for w in words]
print(stemmedWords) """

""" sentence=["hello","how","are","you"]
words=["hello","i","how","thank","do","are","cool","bye","you"]
bag=bagOfWords(sentence,words)
print(bag) """