import pytest
from ..lr_model.config.core import config

@pytest.fixture()
def sample_data_path():
    return config.app_config.sample_test_file
