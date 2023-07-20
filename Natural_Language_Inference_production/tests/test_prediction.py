from ..lr_model.test import run_test
import pytest

def test_prediction(sample_data_path):
    output = run_test(sample_data_path)
    for pred in output:
        assert pred in ['neutral','entailment','contradiction']
