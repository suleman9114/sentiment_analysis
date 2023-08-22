#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:19:52 2023

@author: soleman
"""


from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class TrainTextAnalyzeModel:

    def __init__(self, top_words=10000, max_review_length=500, emb_vect_length=32):

        # Load IMDb dataset
        # Only keep top_words most frequent words, rest will be discarded
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=top_words)
        
        # Pad sequences to max_review_length
        self.x_train = pad_sequences(self.x_train, maxlen=max_review_length)
        self.x_test = pad_sequences(self.x_test, maxlen=max_review_length)

        # Define the model
        model = Sequential()
        model.add(Embedding(top_words, emb_vect_length, input_length=max_review_length))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        self.model = model

    def train_model(self, epochs=10000, batch_size=128):
        
        # Early stopping when validation loss is not improving
        es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        
        # Training the model
        hist = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                              epochs=epochs, batch_size=batch_size, callbacks=[es])
                              
        return hist.history

    def evaluate_model(self):
        
        # Evaluate model performance on training and testing data
        _, train_acc = self.model.evaluate(self.x_train, self.y_train)
        _, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print('Train Acc: %.3f, Test Acc: %.3f' % (train_acc, test_acc))
            
    def save_model(self, model_name):
        # Save the model
        self.model.save(model_name)
        
    def visualize_acc_loss(self, hist):
      
        # Function to visualize training history - Accuracy & Loss over epochs
        plt.figure(figsize=(12, 10))

        # Plot Accuracy
        plt.subplot(2, 1, 1)  
        plt.plot(hist['accuracy'], '-', label='Training')
        plt.plot(hist['val_accuracy'], ':', label='Validation')

        # Plot Loss
        plt.subplot(2, 1, 2)
        plt.plot(hist['loss'], '-', label='Training')
        plt.plot(hist['val_loss'], ':', label='Validation')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ttam = TrainTextAnalyzeModel()
    history = ttam.train_model()
    ttam.evaluate_model()
    ttam.save_model("model_name.h5")
    ttam.visualize_acc_loss(history)