from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# we don't have classification_model folder in our project directory, but we can import module from classification_model folder,
# because we package our model built in library named tsp-classification-model,
# witch was installed in the project virtual environment. So with this manipulation,
# we need to write correctly the path of import like in the packaging model step,
# and it will be recognized automatically

from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent))  # this command beginning in the folder of file prepare_fastapi_api.py everywhere you launch it, this path is in the folder section-production-model-package

from classification_model.configuration import core
from classification_model.processing import data_manager

from building_fastapi_api import create_fastapi_api


# @pytest.fixture: This is a decorator provided by the Pytest framework for defining fixtures.
# Fixtures are Python functions marked with this decorator.
# scope="module": This parameter specifies the scope of the fixture.
# In this case, scope="module" means that the fixture will have module-level scope.
# This means that the fixture function will be executed once per test module (i.e., once for all test functions within a single test module)
# and will be shared among all the tests in that module.
@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return data_manager.load_dataset(file_name=core.config.app_config.test_data_file)


# to create a "Request body" on FastAPI application interface for making HTTP requests to API,
# and then optionally clears any dependency overrides after the test function has finished using the client
@pytest.fixture()
def client() -> Generator:
    with TestClient(create_fastapi_api.api) as _client:  # The TestClient is used for making HTTP requests to your FastAPI application. It allows you to simulate HTTP requests for testing purposes.
        yield _client  # returns _client to the test function that uses this fixture. The "yield" statement temporarily suspends the function's execution and provides "_client" to the caller. This allows the test function to make HTTP requests to the FastAPI application using _client.
        create_fastapi_api.api.dependency_overrides = {}  # to reset any custom dependency overrides that may have been set for the application. Dependency overrides are used in FastAPI to replace dependencies with custom implementations during testing.
