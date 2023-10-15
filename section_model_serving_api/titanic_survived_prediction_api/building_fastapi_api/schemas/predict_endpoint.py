from typing import Any, List, Optional
from pydantic import BaseModel

# sometimes, we don't have classification_model folder in our project directory, but we can import module from classification_model folder,
# because we package our model built in library named tsp-classification-model,
# witch was installed in the project virtual environment. So with this manipulation,
# we need to write correctly the path of import like in the packaging model step,
# and it will be recognized automatically
# It's not also necessary to make the following code to be in the folder witch contains classification_model folder,
# because we don't have classification_model folder in this part of project

from pathlib import Path
import sys
sys.path.append(
    str(Path().absolute().parent.parent))  # this command beginning in the folder of file prepare_fastapi_app.py everywhere you launch it, this path is in the folder section-production-model-package

from classification_model.processing import new_predict_data_preparation


# to specify config type of elements witch will be in the prediction of API
class PredictionResultsPostEndpoint(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


# to specify dictionary of data the api will set or receive in order to provide prediction as result
class MultipleTitanicDataInputs(BaseModel):
    inputs: List[new_predict_data_preparation.TitanicDataInputSchema]

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                        "pclass": 20,
                        "name": "Allen, Miss. Elisabeth Walton",
                        "sex": "female",
                        "age": 30,
                        "sibsp": 2,
                        "parch": 2,
                        "ticket": 24160,
                        "fare": 151.55,
                        "cabin": "C22 C26",
                        "embarked": "S",
                        "boat": 11,
                        "body": 135,
                        "homedest": "New York, NY"
                    }
                ]
            }
        }
