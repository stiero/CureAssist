#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:31:26 2018

@author: henson
"""

import re
import os
os.chdir("/home/henson/Desktop/conditions")

from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer

stop = set(stopwords.words('english'))

tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()


def file_read(file):
    fp = open(file, "r")

    lines = [tkr.tokenize(line.lower()) for line in fp.readlines()]
    
    #lines = [line for sublist in lines for line in sublist]
    
    #lines = [stemmer.stem(line) for sublist in lines for line in sublist]
    
    #lines = [line for sublist in lines for line in sublist]
    
    #lines = list(set(lines))

    regex = re.compile('([^\s\w]|_)+')

    cleaned_lines = []

    for line in lines:
        line_temp = regex.sub('', str(line))
        line_temp = tkr.tokenize(line_temp)
        #line_temp = line_temp.lower()
        cleaned_lines.append(line_temp)

    cleaned_lines = list(filter(None, cleaned_lines))
    
    return cleaned_lines
    #return lines





    
file_list = ["fever.txt", "asthma.txt", "chronic_pain.txt", "cold.txt", "cramps.txt", 
             "depression.txt", "diarrhea.txt", "dizziness.txt", "fatigue.txt", 
             "headache.txt", "hypertension.txt", "nausea.txt", "rash.txt",
             "swelling.txt", "sleepiness.txt"]    



full_text = []

for file in file_list:
    text = file_read(file)
    #print(text)
    full_text.append(text)
    

full_text = [item for sublist in full_text for item in sublist]

vector_size = 500
window_size=5

word2vec = Word2Vec(sentences=full_text,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=100,
                    seed=1000,
                    )

#Words most similar to fever
word2vec.most_similar("fever")


#Some similarities between conditions
print(word2vec.wv.similarity("fever", "asthma"))
print(word2vec.wv.similarity("fever", "pain"))
print(word2vec.wv.similarity("cold", "fever"))
print(word2vec.wv.similarity("cold", "headache"))
print(word2vec.wv.similarity("fever", "nausea"))






    
    
