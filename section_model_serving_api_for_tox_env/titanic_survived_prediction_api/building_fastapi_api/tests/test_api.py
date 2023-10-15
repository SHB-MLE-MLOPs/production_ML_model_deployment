import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    # use client create for making simulation of HTTP requests to API and initialized with api witch exist in create_fastapi_api.py,
    # and play data on the POST endpoint of api through client created (data witch is loaded in previous "payload").
    # This provides "Request body" space on the interface of FastAPI application and customer can make request to API in this space
    # The "Request body" space will be on the POST endpoint
    response = client.post(
        "http://localhost:8001/api/titanic.survived.predict/v1/prediction",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] is None
    assert (math.isclose(prediction_data["predictions"][0], 1, abs_tol=0.0001) or
            math.isclose(prediction_data["predictions"][0], 0, abs_tol=0.0001))
