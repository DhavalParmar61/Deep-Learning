import pytest

from lr_model.config.core import config


@pytest.fixture()
def sample_data_path():
    return config.app_config.validation_data_file


@pytest.fixture()
def sample_input():
    return {
        "sentences1": [
            "This church choir sings to the masses as they sing joyous songs from the book at a church.",
            "This church choir sings to the masses as they sing joyous songs from the book at a church.",
        ],
        "sentences2": [
            "The church has cracks in the ceiling.",
            "The church is filled with song.",
        ],
    }
