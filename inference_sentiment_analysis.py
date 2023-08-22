#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:19:52 2023

@author: soleman
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import string
import numpy as np
import matplotlib.pyplot as plt

class SentimentInference:

    def __init__(self, top_words=1000, max_review_len=500):
        
        # Initialize variables 
        self.top_words = top_words
        self.max_review_length = max_review_len
        
        # Load model
        try:
            self.model = load_model('model_name.h5')
        except Exception as e:
            print("Model could not be loaded. Error: ", e)
            return None
            
        # Get word index
        word_dict = imdb.get_word_index()
        word_dict = {key: (value + 3) for key, value in word_dict.items()}
        word_dict[''] = 0  # padding
        word_dict['>'] = 1  # start
        word_dict['?'] = 2  # unknown/oov
        self.word_dict = word_dict

    def get_result(self, text):
        
        # Pre-process the input
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = text.lower().split(' ')
        text = [word for word in text if word.isalpha()]
        
        # Convert words to integers
        input = [1]
        for word in text:
            if word in self.word_dict:
                if self.word_dict[word] < self.top_words:
                    input.append(self.word_dict[word])
                else:
                    input.append(2)
            
        # Padding the input
        padded_input = pad_sequences([input], maxlen=self.max_review_length)
        
        # Predict the sentiment using the model
        try:
            pred = self.model.predict(np.array([padded_input][0]))[0][0]
        except Exception as e:
            print("Prediction error: ", e)
            return None
        
        # Setting a threshold for classifying the sentiment
        if pred > 0.5:
            return "Positive"
        else:
            return "Negative"

# Predict sentiment of a given text
text = input("Enter your review here: ")
si = SentimentInference()
result = si.get_result(text)
print("Sentiment of Review: ", result)