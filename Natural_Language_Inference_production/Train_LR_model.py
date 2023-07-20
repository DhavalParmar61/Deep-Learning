import string
import pickle
import numpy as np
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack, hstack
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def preprocess(filename):
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

#Golden Label to integer
def GL_to_int(GL):
    for i in range(len(GL)):
        if GL[i] == 'neutral':
            GL[i] = 0
        elif GL[i] == 'entailment':
            GL[i] = 1
        elif GL[i] == 'contradiction':
            GL[i] = 2

    GL = np.array(GL)
    return(GL)

if __name__ == "__main__":

    [corpus_train, Y_train] = preprocess('snli_1.0_train.jsonl')
    Y_train = GL_to_int(Y_train)

    corpus1_train = []
    corpus2_train = []
    for i in range(len(corpus_train)//2):
        corpus1_train.append(corpus_train[2*i])
        corpus2_train.append(corpus_train[2*i+1])

    vectorizer_s1 = TfidfVectorizer()
    s1_train = vectorizer_s1.fit_transform(corpus1_train)
    f1 = open('feature_s1.pkl','wb')
    pickle.dump(vectorizer_s1,f1)
    f1.close

    vectorizer_s2 = TfidfVectorizer()
    s2_train = vectorizer_s2.fit_transform(corpus2_train)
    f2 = open('feature_s2.pkl','wb')
    pickle.dump(vectorizer_s2,f2)
    f2.close

    X_train = hstack([s1_train,s2_train])

    # Classifier
    model = LogisticRegression(max_iter = 5000)
    model.fit(X_train, Y_train)
    pickle.dump(model,open('./models/LR_model.sav','wb'))