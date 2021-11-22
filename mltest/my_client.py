import io
import uuid

import cloudpickle
import numpy as np
import requests
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mltest.clients import minio_client


class Client(object):

    def __init__(self, backend_url='http://127.0.0.1:8000'):
        self.backend_url = backend_url

        self.minio_client = minio_client

    def test_random_forest(self, random_forest_model, x_test, y_test, metadata):
        pickled_model = io.BytesIO()
        cloudpickle.dump(random_forest_model, pickled_model)
        pickled_model.seek(0)

        pickled_data = io.BytesIO()

        np.savez_compressed(pickled_data, x_test=x_test, y_test=y_test)
        pickled_data.seek(0)

        job_uuid = uuid.uuid4()

        # Upload this files to the S3 servera
        # self, bucket_name, object_name, data, length,
        minio_client.put_object('jobs', f'{job_uuid}.model', pickled_model, len(pickled_model.getvalue()))
        minio_client.put_object('jobs', f'{job_uuid}.data', pickled_data, len(pickled_data.getvalue()))

        request_url = self.backend_url + '/sdreipoint'

        response = requests.post(request_url, json={'job_id': str(job_uuid)})

        print(response)


client = Client()

x, y = make_classification(n_classes=3, n_informative=10)
x_train, x_test, y_train, y_test = train_test_split(x, y)

rf = RandomForestClassifier(n_estimators=4)

rf.fit(x_train, y_train)

client.test_random_forest(rf, x_test, y_test, {})
