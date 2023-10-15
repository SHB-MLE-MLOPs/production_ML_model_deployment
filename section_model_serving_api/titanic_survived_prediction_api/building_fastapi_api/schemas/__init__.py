from pathlib import Path
import sys

sys.path.append(
    str(Path().absolute().parent))  # this command beginning in the folder of file prepare__init__.py everywhere you launch it, this path is in the folder section-production-model-package

from schemas.settings_endpoint import SettingsGetEndpoint
from schemas.predict_endpoint import MultipleTitanicDataInputs, PredictionResultsPostEndpoint
