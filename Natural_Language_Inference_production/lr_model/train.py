import pickle

from config.core import PACKAGE_ROOT, config
from processing.dataprocessing import get_data
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def run_training() -> None:
    [corpus_train, Y_train] = get_data(config.app_config.training_data_file)

    corpus1_train = []
    corpus2_train = []
    for i in range(len(corpus_train) // 2):
        corpus1_train.append(corpus_train[2 * i])
        corpus2_train.append(corpus_train[2 * i + 1])

    vectorizer_s1 = TfidfVectorizer()
    s1_train = vectorizer_s1.fit_transform(corpus1_train)
    with open(PACKAGE_ROOT / config.app_config.sentence1_vectorizer, "wb") as f1:
        pickle.dump(vectorizer_s1, f1)

    vectorizer_s2 = TfidfVectorizer()
    s2_train = vectorizer_s2.fit_transform(corpus2_train)
    with open(PACKAGE_ROOT / config.app_config.sentence2_vectorizer, "wb") as f2:
        pickle.dump(vectorizer_s2, f2)

    X_train = hstack([s1_train, s2_train])

    # Classifier
    model = LogisticRegression(max_iter=config.lr_info_config.max_iterations)
    model.fit(X_train, Y_train)
    with open(
        PACKAGE_ROOT / config.lr_info_config.output_model_path, "wb"
    ) as model_file:
        pickle.dump(model, model_file)


if __name__ == "__main__":
    run_training()
