from config.core import config
from processing.dataprocessing import get_data, int_to_GL
import pickle
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

def run_test() -> None:
    [corpus_test, GL_test] = get_data(config.app_config.test_data_file)

    # Logistic Regression model
    corpus1_test = []
    corpus2_test = []
    for i in range(len(corpus_test) // 2):
        corpus1_test.append(corpus_test[2 * i])
        corpus2_test.append(corpus_test[2 * i + 1])

    vectorizer1 = pickle.load(open(config.app_config.sentence1_vectorizer, 'rb'))
    s1_test = vectorizer1.transform(corpus1_test)
    vectorizer2 = pickle.load(open(config.app_config.sentence2_vectorizer, 'rb'))
    s2_test = vectorizer2.transform(corpus2_test)

    X_test = hstack([s1_test, s2_test])

    # Classifier
    model_LR = LogisticRegression(max_iter=config.model_config.max_iterations)
    model_LR = pickle.load(open(config.model_config.model_path, 'rb'))
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

    with open(config.model_config.output_path, 'w') as f:
        for i in range(len(output)):
            f.write(f"%s\n" % output[i])

if __name__=="__main__":
    run_test()