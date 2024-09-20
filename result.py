# Gestione degli argomenti
import argparse

parser = argparse.ArgumentParser(
                    prog='LIME Text Model Explainer',
                    description='Explain using the LIME tool a model prediction of a Movie Review [Positive or Negative]')

parser.add_argument('--m', '--model', dest='model_file', required = True,
                    help='The name of the model to use')
parser.add_argument('--o', '--output', dest='output_file', required = False,
                    help='The name of the output file to use')
parser.add_argument('--t', '--text', dest='text_input', required = True,
                    help='The text input to be used')

args = parser.parse_args()

# Carico il modello
import keras

model = keras.saving.load_model(args.model_file)

"""# Setup LIME

## Funzioni utili
"""

from keras.datasets import imdb

# Load the IMDb dataset's word index (mapping from words to integer indices)
word_index = imdb.get_word_index()

# Indices 1 and 2 are reserved for special tokens
# 1 -> "start of sequence", 2 -> "padding"
INDEX_FROM = 3  # We will offset by 3 to account for these reserved tokens

# Invert the word index to map integer indices back to words
reverse_word_index = {value + INDEX_FROM: key for (key, value) in word_index.items()}

# Function to decode a sequence back to text
def decode_sequence(sequence):
    # Indices start from 3 (1 and 2 are reserved)
    return ' '.join([reverse_word_index.get(i, '?') for i in sequence])

# Special token for out-of-vocabulary (OOV) words
OOV_TOKEN = 3  # Maps to index 3

# Function to encode a list of texts into sequences of integers
def encode_sequences(texts):
    encoded_sequences = []

    # Iterate over each text in the list
    for text in texts:
        # Ensure the text is in lowercase and split into words
        words = text.lower().split()

        # Map each word to the corresponding integer index in the word_index
        # If a word is not found in the word_index, map it to the OOV token
        sequence = [word_index.get(word, OOV_TOKEN) + INDEX_FROM for word in words]

        encoded_sequences.append(sequence)

    return encoded_sequences

"""## Configurazione"""

import numpy as np
import lime

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

class CustomVectorizer(BaseEstimator, TransformerMixin):
  def __init__(self, word_index):
    # Store the word index, which maps words to integer indices
    self.word_index = word_index

  def fit(self, X, y=None):
    # No fitting needed for this vectorizer, just return self
    return self

  def transform(self, X):
    return encode_sequences(X)

# Create an instance of the CustomVectorizer, passing the word_index
custom_vectorizer = CustomVectorizer(word_index=word_index)

# Create a pipeline with the custom vectorizer and your model
pipeline = make_pipeline(custom_vectorizer, model)

class_names = ['Negative', 'Positive']
# Create a LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

def predict_fn(texts):
  # Tokenize the raw text using the custom vectorizer
  tokenized_texts = custom_vectorizer.transform(texts)

  # Pad all sequences to the same length
  max_sequence_length = maxlen
  padded_sequences = preprocessing.sequence.pad_sequences(tokenized_texts, maxlen=max_sequence_length)

  # Use the model to predict the class probabilities
  predictions = model.predict(padded_sequences)

  # Since it's a binary classification, convert the single output to two probabilities
  predictions = np.hstack((1 - predictions, predictions))

  return predictions

"""# Mostro le predizioni"""

import time
from keras import preprocessing

maxlen = 100

exp = explainer.explain_instance(args.text_input, predict_fn, num_features=10, labels=[0])

if(args.output_file == None):
  outputFileName = "output.html"
else:
  outputFileName = args.output_file


# Write the output file
from pathlib import Path

output_file = Path("Output/" + outputFileName)
output_file.parent.mkdir(exist_ok=True, parents=True)
exp.save_to_file("Output/" + outputFileName)

# Open the result on the browser
import webbrowser
import os

cwd = os.getcwd()
webbrowser.open(cwd + "/Output/" + outputFileName)