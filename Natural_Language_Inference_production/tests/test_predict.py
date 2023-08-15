from lr_model.predict import make_prediction


def test_prediction(sample_input):
    result = make_prediction(sample_input)
    assert result["errors"] == None
    for prediction in result["predictions"]:
        assert prediction in ["neutral", "entailment", "contradiction"]
