#!/usr/bin/env python
# coding: utf-8

# # Hate Speech Detection Using Machine Learning
# 

# In[2]:


import pandas as pd  
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
from pandas import *

#clean_string(): takes a string removes not alphabet characters and stopwords

#clean_dataset(): takes a csv filename and cleans it returns data sets and labels

def clean_string(raw_review):
    
    example1 = BeautifulSoup(raw_review) 

    example1 = re.sub('[^a-zA-Z]'," ",example1.get_text())

    words = example1.lower().split()
    
    stops = set(stopwords.words('english'))
    
    words = [w for w in words if not w in stops] 
    
    
    sentence = " ".join(words)
    
    return sentence


def clean_dataset(filename):
    
    with open(filename, 'r') as dat:
        lines = dat.readlines()
    data_set = []
    data_labels = []
    for i in range(1,len(lines)):
        if len(lines[i].split(',')) < 6 or not lines[i].split(',')[5].isdigit():
            continue
        if int(lines[i].split(',')[5]) == 0: # hatespeech
            data_labels.append(1)
            data_set.append(clean_string(lines[i])) 
        elif int(lines[i].split(',')[5]) == 2: # not hatespeech
            data_labels.append(0)
            data_set.append(clean_string(lines[i])) 

    return data_set,data_labels

def main():    
    filename = 'labeled_data.csv'

    data_set,data_labels = clean_dataset(filename)

    # create a csv file of the clean dataset

    df = DataFrame({'tweet': data_set, 'class': data_labels})
    df.to_csv('data.csv', sep='\t')



main()


# In[ ]:




