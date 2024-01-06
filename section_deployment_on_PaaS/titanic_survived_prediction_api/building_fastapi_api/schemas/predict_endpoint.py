from typing import Any, List, Optional
from pydantic import BaseModel
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
