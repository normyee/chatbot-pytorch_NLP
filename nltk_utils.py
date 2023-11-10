import nltk
import numpy as np

# Uncomment the line below if the pre-trained tokenizer is not downloaded
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# preprocessing data

# tokenize data

def tokenize(sentence: str) -> list:
    return nltk.word_tokenize(sentence)

# stem data in lower case

def stem(word: str) -> list:
    return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    BoW   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


# example below on how it works
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bow = bag_of_words(sentence, words)

# print(bow)
#output should be: [0. 1. 0. 1. 0. 0. 0.]