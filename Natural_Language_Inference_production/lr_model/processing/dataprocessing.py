import json
import string

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))


def preprocess_sentence(sentence):
    # Convert to lower case
    sentence = sentence.lower()

    # Removing punctuation mark
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))

    # Removing stop words
    tokens = word_tokenize(sentence)
    sentence = [i for i in tokens if i not in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    sentence = " ".join([lemmatizer.lemmatize(w) for w in sentence])
    return sentence


def get_data(filename):
    # Reading Training Data
    print("Reading data")
    f = open(filename, "r")
    Gold_label = []
    sentence_a = []
    sentence_b = []
    for line in f:
        data = json.loads(line)
        if data["gold_label"] != "-":
            Gold_label.append(data["gold_label"])
            sentence_a.append(data["sentence1"])
            sentence_b.append(data["sentence2"])
        else:
            a = 0
            b = 0
            c = 0
            GL_anotator = data["annotator_labels"]
            for i in range(len(GL_anotator)):
                if GL_anotator[i] == "neutral":
                    a = a + 1
                elif GL_anotator[i] == "contradiction":
                    b = b + 1
                elif GL_anotator[i] == "entailment":
                    c = c + 1
            m = max(a, b, c)
            count = 0
            if m == a:
                Gold_label.append("neutral")
                count += 1
            if m == b:
                Gold_label.append("contradiction")
                count += 1
            if m == c:
                Gold_label.append("entailment")
                count += 1
            for i in range(count):
                sentence_a.append(data["sentence1"])
                sentence_b.append(data["sentence2"])
    f.close()

    print("preprocessing text sentences")
    corpus = []
    for n in range(len(sentence_a)):
        # Preprocessing
        corpus.append(preprocess_sentence(sentence_a[n]))
        corpus.append(preprocess_sentence(sentence_b[n]))

    print("Concerting Gold label to integer class")
    GL_list = []
    for GL in Gold_label:
        GL_list.append(GL_to_int(GL))

    Gold_label = np.array(GL_list)
    return [corpus, Gold_label]


def preprocess_input(input):
    if isinstance(input, dict):
        if set(["sentences1", "sentences2"]).issubset(input.keys()):
            if isinstance(input["sentences1"], list) and isinstance(
                input["sentences1"], list
            ):
                s1_list = input["sentences1"]
                s2_list = input["sentences2"]
                if len(s1_list) == len(s2_list):
                    processed_s1 = []
                    processed_s2 = []
                    for i in range(len(s1_list)):
                        if not (
                            isinstance(s1_list[i], str) and isinstance(s2_list[i], str)
                        ):
                            return None, "Sentences must be of string type"
                        processed_s1.append(preprocess_sentence(s1_list[i]))
                        processed_s2.append(preprocess_sentence((s2_list[i])))
                    return [processed_s1, processed_s2], None
                else:
                    return None, "Both sentence lists should be of same length"
            else:
                return None, "Value for Sentences1 and Sentences2 should list"
        else:
            return None, "sentence1 or sentence2 key missing"
    else:
        return None, "Input should be a dictionary"


# Golden Label to integer
def GL_to_int(GL):
    if GL == "neutral":
        GL = 0
    elif GL == "entailment":
        GL = 1
    elif GL == "contradiction":
        GL = 2
    else:
        raise Exception("Not a valid target class")

    return GL


# integer to GL
def int_to_GL(GL):
    if GL == 0:
        GL = "neutral"
    elif GL == 1:
        GL = "entailment"
    elif GL == 2:
        GL = "contradiction"
    return GL
