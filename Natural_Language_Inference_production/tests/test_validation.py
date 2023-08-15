from lr_model.run_validation import run_validation


def test_prediction(sample_data_path):
    output = run_validation(sample_data_path)
    for pred in output:
        assert pred in ["neutral", "entailment", "contradiction"]
