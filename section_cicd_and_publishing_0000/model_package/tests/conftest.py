import pytest
import pandas as pd

from pprint36 import pprint
from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent))  # this command beginning in the folder of file conftest.py everywhere you launch it, this path is in the folder section-production-model-package
from classification_model.configuration import core
from classification_model.processing import data_manager


@pytest.fixture()
def sample_input_data() -> pd.DataFrame:
    return data_manager.load_dataset(file_name=core.config.app_config.test_data_file)
