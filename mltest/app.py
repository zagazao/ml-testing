import io
import uuid

import cloudpickle
import numpy as np
import sklearn
from fastapi import FastAPI, File, Depends
from pydantic import BaseModel

from mltest.celery_app import celery_app
from mltest.some_tests import generate_test_pipeline

app = FastAPI()

rf_test_pipeline = generate_test_pipeline()


class TestRequest(BaseModel):
    model: sklearn.base.ClassifierMixin
    features: np.ndarray
    targets: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class TestRfRequest(BaseModel):
    job_id: str


class TestRfResponse(BaseModel):
    job_id: str


def model_and_data(model: bytes = File(...),
                   data: bytes = File(...)) -> TestRequest:
    """
    Dependency injection utility for API endpoints.
    :param model:
    :param data:
    :return:
    """
    # Recover model
    model_bytes_io = io.BytesIO(model)
    recovered_model = cloudpickle.load(model_bytes_io)

    # Recover data
    data_bytes_io = io.BytesIO(data)
    recovered_data = np.load(data_bytes_io)

    return TestRequest(
        model=recovered_model,
        features=recovered_data['x_test'],
        targets=recovered_data['y_test']
    )


@app.post("/sdreipoint/")
async def create_file(request: TestRfRequest):
    # TODO: Submit to celery...

    celery_app.send_task('test_random_forest_task',
                         args=(request.job_id,),
                         task_id=request.job_id)

    return TestRfResponse(job_id=request.job_id)


@app.get("/status/")
def get_task_status(task_id: str):
    res = celery_app.AsyncResult(task_id).state
    return res


@app.post("/files/")
async def create_file(test_request: TestRequest = Depends(model_and_data)):
    # TODO: Submit to celery...

    _id = uuid.uuid4()

    # x, y = make_classification(n_classes=3, n_informative=10)
    # x_train, x_test, y_train, y_test = train_test_split(x, y)

    result = rf_test_pipeline.run(test_request.model, test_request.features, test_request.targets)

    print(result.text)

    return result
