from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lr_model.config.core import config

from app.main import app


@pytest.fixture(scope="module")
def test_data() :
    input = {
                "sentences1": [
                    "This church choir sings to the masses as they sing joyous songs from the book at a church.",
                    "This church choir sings to the masses as they sing joyous songs from the book at a church.",
                ],
                "sentences2": [
                    "The church has cracks in the ceiling.",
                    "The church is filled with song.",
                ]
            }

    return input


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
