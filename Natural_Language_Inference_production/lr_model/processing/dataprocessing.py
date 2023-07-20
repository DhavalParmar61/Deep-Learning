import string
import pickle
import numpy as np
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_sentence(sentence):
    #Convert to lower case
    sentence = sentence.lower()

    # Removing punctuation mark
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))

    # Removing stop words
    tokens = word_tokenize(sentence)
    sentence = [i for i in tokens if not i in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    sentence = " ".join([lemmatizer.lemmatize(w) for w in sentence])
    return sentence

def get_data(filename):
    # Reading Training Data
    f = open(filename, 'r')
    Gold_label = []
    sentence_a = []
    sentence_b = []
    for line in f:
        data = json.loads(line)
        if data["gold_label"] != '-':
            Gold_label.append(data["gold_label"])
            sentence_a.append(data["sentence1"])
            sentence_b.append(data["sentence2"])
        else :
            a=0;b=0;c=0;
            GL_anotator = data["annotator_labels"]
            for i in  range(len(GL_anotator)):
                if GL_anotator[i] == 'neutral':
                    a = a+1;
                elif GL_anotator[i] == 'contradiction':
                    b = b+1;
                elif GL_anotator[i] == 'entailment':
                    c = c+1;
            m = max(a,b,c)
            count = 0
            if m==a:
                Gold_label.append('neutral')
                count +=1
            if m==b:
                Gold_label.append('contradiction')
                count += 1
            if m==c:
                Gold_label.append('entailment')
                count += 1
            for i in range(count):
                sentence_a.append(data["sentence1"])
                sentence_b.append(data["sentence2"])
    f.close()

    corpus = []
    for n in range(len(sentence_a)):
        # Preprocessing
        corpus.append(preprocess_sentence(sentence_a[n]))
        corpus.append(preprocess_sentence(sentence_b[n]))

    return [corpus, Gold_label]

#Golden Label to integer
def GL_to_int(GL):
    for i in range(len(GL)):
        if GL[i] == 'neutral':
            GL[i] = 0
        elif GL[i] == 'entailment':
            GL[i] = 1
        elif GL[i] == 'contradiction':
            GL[i] = 2
        else:
            raise Exception("Not a valid target class")

    GL = np.array(GL)
    return(GL)

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