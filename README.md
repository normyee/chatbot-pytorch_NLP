# ChatBot with Natural Language Processing

![image](https://github.com/normyee/chatbot-pytorch_NLP/assets/63208510/8edcef31-84fc-4995-808b-7254dd203d9e)
## Overview
- This project implements a ChatBot using a neural network in Python, leveraging Natural Language Processing (NLP) techniques, Bag of Words model, and preprocessing methods like tokenization and stemming.

## Technologies Used
- NLTK: Natural Language Toolkit for various NLP tasks.
  
- PyTorch: Deep learning library for building and training neural networks.
  
- NumPy: Fundamental package for scientific computing with Python.

## Project Structure
- `train.py`: Python script for training the neural network with the provided data.

- `chat.py`: Python script to initialize and run the ChatBot.

- `model.py`: Contains the feedforward neural network model with two hidden layers.

- `nltk_utils.py`: Includes functions for preprocessing the data.

- `intents.json`: Document containing the training data.

# How to Use
## Training the Model

```
python train.py
```

## Initializing the ChatBot
To start the ChatBot, use the following command:

```
python chat.py
```

# Dependencies
Make sure to install the necessary dependencies by running:

```
pip install nltk torch numpy 
```
# Notes
- Ensure that the NLTK data is downloaded before running the scripts: `uncomment # nltk.download('punkt')`


