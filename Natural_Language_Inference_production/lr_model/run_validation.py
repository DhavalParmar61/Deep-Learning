import pickle

import numpy as np
from scipy.sparse import hstack

from lr_model.config.core import config
from lr_model.processing.dataprocessing import get_data, int_to_GL


def run_validation(test_path) -> None:
    [corpus_test, GL_test] = get_data(test_path)

    # Logistic Regression model
    corpus1_test = []
    corpus2_test = []
    for i in range(len(corpus_test) // 2):
        corpus1_test.append(corpus_test[2 * i])
        corpus2_test.append(corpus_test[2 * i + 1])

    with open(config.app_config.sentence1_vectorizer, "rb") as s1_vectorizer:
        vectorizer1 = pickle.load(s1_vectorizer)
    s1_test = vectorizer1.transform(corpus1_test)

    with open(config.app_config.sentence2_vectorizer, "rb") as s2_vectorizer:
        vectorizer2 = pickle.load(s2_vectorizer)
    s2_test = vectorizer2.transform(corpus2_test)

    X_test = hstack([s1_test, s2_test])

    # Classifier
    with open(config.lr_info_config.output_model_path, "rb") as model:
        model_LR = pickle.load(model)
    output = model_LR.predict(X_test)
    output = np.ndarray.tolist(output)

    # Evaluating Prediction
    N = 0
    correct_label = 0
    for i in range(len(output)):
        if (GL_test[i]) != "-":
            N += 1
            if GL_test[i] == output[i]:
                correct_label += 1

    acc = correct_label / N
    print("Accuracy on validation data for LR model : %f" % acc)

    # Converting prediction to label
    for i in range(len(output)):
        output[i] = int_to_GL(output[i])

    with open(config.lr_info_config.output_path, "w") as f:
        for i in range(len(output)):
            f.write("%s\n" % output[i])

    return output


if __name__ == "__main__":
    run_validation(config.app_config.validation_data_file)
