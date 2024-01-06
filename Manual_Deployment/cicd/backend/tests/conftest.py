import pandas as pd
import pytest

from classification_model.configuration import core
from classification_model.processing import data_manager


@pytest.fixture()
def sample_input_data() -> pd.DataFrame:
    return data_manager.load_dataset(file_name=core.config.app_config.test_data_file)
