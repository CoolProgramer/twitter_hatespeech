#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy
import nltk
import pickle

def tokenize(sentences):
    words = []
    for sentence in sentences:
        try:
            w = sentence.split(' ')
            words.extend(w)
        except:
            pass
    words = sorted(list(set(words)))
    return words



def reduce_vocab_size(vocab):
    dictionary = set(nltk.corpus.words.words())
    vocab = [w for w in vocab if w in dictionary] 
    return vocab


def vectorize(sentence,vocab): 
    

    words = sentence.split(' ')
    bag_vector = numpy.zeros(len(vocab))
    for w in words:
        for i,word in enumerate(vocab):
            if word == w: 
                bag_vector[i] += 1

    return bag_vector

def vectorize_dataset(tweets,labels):
    vocab = tokenize(tweets)
    vocab = reduce_vocab_size(vocab)
    v = []
    u = []
    for i in range(len(tweets)):
        try:
            v.append(vectorize(tweets[i],vocab))
            u.append(labels[i])
        except:
            pass
    return v,u

def main():
    
    df=pd.read_csv('data.csv',delimiter='\t',encoding='utf-8')

    tweets = list(df['tweet'])
    datalabels = list(df['class'])

    dataset,datalabels = vectorize_dataset(tweets,datalabels)
    file = open('data.pickle', 'wb')

    # dump information to that file
    pickle.dump([dataset,datalabels], file)

    # close the file
    file.close()
main()


# In[ ]:




