import nltk

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
    pass