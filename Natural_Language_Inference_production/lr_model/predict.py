import pickle

import numpy as np
from scipy.sparse import hstack

from lr_model import __version__ as _version
from lr_model.config.core import PACKAGE_ROOT, config
from lr_model.processing.dataprocessing import int_to_GL, preprocess_input

vectorizer1 = pickle.load(
    open(PACKAGE_ROOT / config.app_config.sentence1_vectorizer, "rb")
)
vectorizer2 = pickle.load(
    open(PACKAGE_ROOT / config.app_config.sentence2_vectorizer, "rb")
)

with open(PACKAGE_ROOT / config.lr_info_config.output_model_path, "rb") as model:
    model_LR = pickle.load(model)


def make_prediction(input: dict) -> dict:
    processed_input, error = preprocess_input(input)
    results = {"predictions": None, "version": _version, "errors": error}

    if not error:
        s1 = vectorizer1.transform(processed_input[0])
        s2 = vectorizer2.transform(processed_input[1])

        X = hstack([s1, s2])

        # Classifier
        output = model_LR.predict(X)
        output = np.ndarray.tolist(output)

        # Converting prediction to label
        for i in range(len(output)):
            output[i] = int_to_GL(output[i])

        results = {"predictions": output, "version": _version, "errors": error}

    return results
