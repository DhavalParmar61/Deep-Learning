import nltk
import pickle
import matplotlib.pyplot as plt
import string
import pandas as pd
import tensorflow.keras.backend as k
import numpy as np
import tensorflow as tf
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Lambda, Embedding, Bidirectional,TimeDistributed,Flatten,Activation,RepeatVector,Permute,Multiply,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def Creat_LSTM_model():
    # Two set of input
    inputA = Input(shape=(None,))
    inputB = Input(shape=(None,))

    embedding_layer = Embedding(input_dim=(len(word_indx) + 1), output_dim=300, weights=[embedding_matrix],
                                mask_zero=True,
                                trainable=False)
    lstm = Bidirectional(LSTM(300, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    # first branch
    u = embedding_layer(inputA)
    x = lstm(u)
    '''
    attention_x = TimeDistributed(Dense(1, activation='tanh'))(x)
    attention_x = Lambda(lambda x: x)(attention_x)
    attention_x = Flatten()(attention_x)
    attention_x = Activation('softmax')(attention_x)
    attention_x = RepeatVector(600)(attention_x)
    attention_x = Permute([2, 1])(attention_x)

    # apply the attention
    sent_representation = Multiply()([x, attention_x])
    x = Lambda(lambda xin: k.sum(xin, axis=1))(sent_representation)
    '''
    # second branch
    v = embedding_layer(inputB)
    y = lstm(v)
    '''
    attention_y = TimeDistributed(Dense(1, activation='tanh'))(y)
    attention_y = Lambda(lambda x: x)(attention_y)
    attention_y = Flatten()(attention_y)
    attention_y = Activation('softmax')(attention_y)
    attention_y = RepeatVector(600)(attention_y)
    attention_y = Permute([2, 1])(attention_y)

    # apply the attention
    sent_representation = Multiply()([y, attention_y])
    y = Lambda(lambda xin: k.sum(xin, axis=1))(sent_representation)
    '''
    c1 = Lambda(lambda a: tf.subtract(a[0], a[1]))([x, y])
    c2 = Lambda(lambda a: tf.multiply(a[0], a[1]))([x, y])
    ry = Lambda(lambda a: k.reverse(a, axes=0))(y)
    c3 = Lambda(lambda a: tf.subtract(a[0], a[1]))([x, ry])
    c4 = Lambda(lambda a: tf.multiply(a[0], a[1]))([x, ry])

    w = Concatenate()([x, c1, c2, c3, c4, y])
    z = Dense(1800, activation="relu")(w)
    z = Dropout(0.5)(z)
    z = Dense(512, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(128, activation="relu")(z)
    z = Dense(3, activation="softmax")(z)
    model_LSTM = Model(inputs=[inputA, inputB], outputs=z)
    return (model_LSTM)

def smoothed_label(array):
  l = len(array)
  GL = np.zeros(3)
  count = dict()
  count['neutral']=0
  count['entailment']=0
  count['contradiction']=0
  for i in array:
    if i == 'neutral':
      count['neutral'] += 1
    elif i == 'entailment':
      count['entailment']+=1
    elif i == 'contradiction':
      count['contradiction']+=1
  GL[0] = count['neutral']/l
  GL[1] = count['entailment']/l
  GL[2] = count['contradiction']/l
  return(GL)

#Golden Label to integer
def GL_to_int(GL):
  for i in range(len(GL)):
    if GL == 'neutral':
      GL = [1,0,0]
    elif GL == 'entailment':
      GL = [0,1,0]
    elif GL == 'contradiction':
      GL = [0,0,1]
  GL = np.asarray(GL)
  return(GL)

def preprocess(filename):
    # Reading Training Data
    f = open(filename, 'r')
    Gold_label = []
    sentence_a = []
    sentence_b = []
    for line in f:
        data = json.loads(line)
        Gold_label.append(smoothed_label(data["annotator_labels"]))
        sentence_a.append(data["sentence1"])
        sentence_b.append(data["sentence2"])
    Gold_label = np.asarray(Gold_label)
    f.close()

    corpus = []
    for n in range(len(sentence_a)):
        # Preprocessing
        # converting to lowercase
        sentence_1 = sentence_a[n].lower()
        sentence_2 = sentence_b[n].lower()

        # Removing punctuation mark
        sentence_1 = sentence_1.translate(str.maketrans("", "", string.punctuation))
        sentence_2 = sentence_2.translate(str.maketrans("", "", string.punctuation))

        # Removing stop words
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(sentence_1)
        sentence_1 = [i for i in tokens if not i in stop_words]
        tokens = word_tokenize(sentence_2)
        sentence_2 = [i for i in tokens if not i in stop_words]

        # lemmatization
        lemmatizer = WordNetLemmatizer()
        sentence_1 = " ".join([lemmatizer.lemmatize(w) for w in sentence_1])
        sentence_2 = " ".join([lemmatizer.lemmatize(w) for w in sentence_2])

        corpus.append(sentence_1)
        corpus.append(sentence_2)

    return [corpus, Gold_label]


if __name__ == "__main__":

    #Training data preprocessing
    [corpus_train, GL_train] = preprocess('snli_1.0_train.jsonl')
    GL_train = GL_to_int(GL_train)

    # Creating dictionary from glove data
    f = open('final_glove.840B.300d.txt', 'r')
    embedding = dict()
    for line in f:
        temp = line.split()
        word = temp[0]
        coefs = np.asarray(temp[1:], dtype='float32')
        embedding[word] = coefs
    f.close()

    # Preparing sequence
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    f = open('tokenizer-300.pickle', 'wb')
    pickle.dump(tokenizer, f)
    f.close()

    sequences = tokenizer.texts_to_sequences(corpus_train)
    sequences = pad_sequences(sequences, padding='post', dtype='int16')
    word_indx = tokenizer.word_index

    # Preparing embedding matrix
    embedding_matrix = np.zeros((len(word_indx) + 1, 300))
    for word, i in word_indx.items():
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_matrix = np.asarray(embedding_matrix)
    np.save('./embedidng_matrix', embedding_matrix)

    # Inputs
    a = []
    b = []
    for i in range((len(sequences) // 2)):
        a.append(sequences[2 * i])
        b.append(sequences[2 * i + 1])
    a = np.array(a)
    b = np.array(b)

    model_LSTM = Creat_LSTM_model()
    model_LSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    with open('model_summary.txt', 'w') as fh:
        model_LSTM.summary(print_fn=lambda x: fh.write(x + '\n'))

    no_epochs = 7
    checkpoint = ModelCheckpoint('LSTM_model.h5', monitor='val_acc', mode='max', save_best_only=True)
    callback_list = [checkpoint]
    history = model_LSTM.fit([a, b], GL_train, epochs=no_epochs, callbacks = callback_list, validation_split=0.2, batch_size=512)

    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title('Training Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()