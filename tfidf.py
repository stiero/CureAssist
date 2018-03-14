#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:52:46 2018

@author: henson
"""

import re
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
#from nltk.tokenize import TweetTokenizer

import numpy as np

os.chdir("/home/henson/Desktop/conditions")

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

stop = set(stopwords.words('english'))



def file_read(file):
    fp = open(file, "r")

    words = [word.strip() for line in fp.readlines() for word in line.split(" ")
            if word not in stop]

    regex = re.compile('[^a-zA-Z]')

    cleaned_words = []

    for word in words:
        word_temp = regex.sub('', word)
        word_temp = word_temp.lower()
        cleaned_words.append(word_temp)

    cleaned_words = list(filter(None, cleaned_words))

    cleaned_words = ' '.join(cleaned_words)
    
    #return file_text
    
    df = pd.DataFrame({'text': cleaned_words},
                      {str(file[:-4])})
    
    #df['condition'] = str(file[:-4])
    
    return df
    
    #vz = vectorizer.fit_transform(list(df['text']))
    
    #return vz

###############################33
    
file_list = ["fever.txt", "asthma.txt", "chronic_pain.txt", "cold.txt", "cramps.txt", 
             "depression.txt", "diarrhea.txt", "dizziness.txt", "fatigue.txt", 
             "headache.txt", "hypertension.txt", "nausea.txt", "rash.txt",
             "swelling.txt", "sleepiness.txt"]


df = pd.DataFrame()

for file in file_list:
    df_temp = file_read(file)
    
    df = df.append(df_temp)
    

def tfidf_vectorizer(dataframe):
    
    vz = vectorizer.fit_transform(list(dataframe['text']))
    return vz

tfidf_matrix = tfidf_vectorizer(df)

tfidf_dataframe = pd.DataFrame(tfidf_matrix.A,
                            columns = vectorizer.get_feature_names(), 
                            index =[str(file[:-4]) for file in file_list])



from sklearn.metrics.pairwise import cosine_similarity

sims = np.empty((15, 15), dtype="float64")

for row in range(tfidf_matrix.shape[0]):
    sim_temp = cosine_similarity(tfidf_matrix[row:row+1], tfidf_matrix)
    sims[row:] = sim_temp
    
    
colnames = []

for file in file_list:
    name = file[:-4]
    colnames.append(name)

sims = pd.DataFrame(sims,
                  index=colnames,
                  columns=colnames)