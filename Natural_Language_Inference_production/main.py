import keras.backend as k
import numpy as np
import tensorflow as tf
import nltk
import string
import pandas as pd
import pickle
import jsons
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack, hstack
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.layers import LSTM, Dense, Input, Concatenate, Lambda, Embedding, Bidirectional,TimeDistributed,Flatten,Activation,RepeatVector,Permute,Multiply,Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack, hstack
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

def preprocess(filename):
    # Reading Training Data
    f = open(filename, 'r')
    Gold_label = []
    sentence_a = []
    sentence_b = []
    for line in f:
        data = json.loads(line)
        Gold_label.append(data["gold_label"])
        sentence_a.append(data["sentence1"])
        sentence_b.append(data["sentence2"])
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


#integer to GL
def int_to_GL(GL):
    GL = np.ndarray.tolist(GL)
    for i in range(len(GL)):
        if GL[i] == 0:
            GL[i] = 'neutral'
        elif GL[i] == 1:
            GL[i] = 'entailment'
        elif GL[i] == 2:
            GL[i] = 'contradiction'
    return(GL)

def label_output(output_LSTM):
    output_LSTM = np.ndarray.tolist(output_LSTM)
    for i in range(len(output_LSTM)):
        ind = output_LSTM[i].index(max(output_LSTM[i]))
        if ind == 0:
            output_LSTM[i] = 'neutral'
        elif ind == 1:
            output_LSTM[i] = 'entailment'
        elif ind == 2:
            output_LSTM[i] = 'contradiction'
    return (output_LSTM)


if __name__ == '__main__':

    [corpus_test, GL_test] = preprocess('snli_1.0_test.jsonl')

    #Logistic Regression model
    corpus1_test = []
    corpus2_test = []
    for i in range(len(corpus_test)//2):
        corpus1_test.append(corpus_test[2*i])
        corpus2_test.append(corpus_test[2*i+1])

    vectorizer1 = pickle.load(open('feature_s1.pkl','rb'))
    s1_test = vectorizer1.transform(corpus1_test)
    vectorizer2 = pickle.load(open('feature_s2.pkl','rb'))
    s2_test = vectorizer2.transform(corpus2_test)

    X_test = hstack([s1_test,s2_test])

    # Classifier
    model_LR = LogisticRegression(max_iter = 5000)
    model_LR = pickle.load(open('./models/LR_model.sav','rb'))
    output = model_LR.predict(X_test)
    output = int_to_GL(output)

    N = 0
    correct_label = 0
    for i in range(len(output)):
        if (GL_test[i]) != '-':
            N += 1
            if GL_test[i] == output[i]:
                correct_label += 1
    acc = correct_label / N
    print("Accuracy on test data for LR model : %f" % acc)

    with open('tfidf.txt','w') as f:
        for i in range(len(output)):
            f.write(f"%s\n"%output[i])


    #LSTM model

    # With tokenizer
    f = open('tokenizer-300.pickle', 'rb')
    tokenizer = pickle.load(f)
    sequences = tokenizer.texts_to_sequences(corpus_test)
    sequences = pad_sequences(sequences, padding='post', dtype='int16')
    f.close()
    word_indx = tokenizer.word_index

    embedding_matrix = np.load('embedding_matrix.npy')

    # Two set of input
    a = []
    b = []
    for i in range((len(sequences) // 2)):
        a.append(sequences[2 * i])
        b.append(sequences[2 * i + 1])
    a = np.array(a)
    b = np.array(b)

    # MODEL
    model_LSTM = Creat_LSTM_model()
    model_LSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_LSTM.load_weights('./models/LSTM_model.h5')
    output_LSTM = model_LSTM.predict([a,b])
    output_LSTM = label_output(output_LSTM)

    N=0
    correct_label=0
    for i in range(len(output_LSTM)):
        if(GL_test[i]) != '-':
            N += 1
            if GL_test[i] == output_LSTM[i]:
                correct_label += 1
    acc = correct_label/N
    print("Accuracy on test data for LSTM model : %f" % acc)

    with open('deep_model.txt', 'w') as f:
        for i in range(len(output_LSTM)):
            f.write(f"%s\n"%output_LSTM[i])