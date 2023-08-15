import pytest

from lr_model.processing.dataprocessing import GL_to_int, int_to_GL


@pytest.mark.parametrize("GL", ["neutral", "entailment", "contradiction", "similar"])
def test_conversion(GL):
    if GL in ["neutral", "entailment", "contradiction"]:
        assert GL == int_to_GL(GL_to_int(GL))
    else:
        with pytest.raises(Exception) as exc:
            GL_to_int(GL)
        assert "Not a valid target class" in str(exc.value)
