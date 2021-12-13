import io

import numpy as np

from mltest.datasets.entities import DatasetPayload


class DatasetEncoder(object):

    def decode(self, payload) -> DatasetPayload:
        raise NotImplementedError

    def encode(self, payload: DatasetPayload) -> io.BytesIO:
        raise NotImplementedError


class NumpyDatasetEncoder(DatasetEncoder):

    def decode(self, payload: io.BytesIO) -> DatasetPayload:
        # noinspection PyTypeChecker
        data = io.BytesIO(payload)

        data = np.load(data)  # data = np.load(io.BytesIO(file.read()))

        return DatasetPayload(x_train=data['x_train'],
                              y_train=data['y_train'],
                              x_test=data['x_test'],
                              y_test=data['y_test'])

    def encode(self, payload) -> io.BytesIO:
        data = io.BytesIO()
        np.savez_compressed(data,
                            x_train=payload.x_train,
                            y_train=payload.y_train,
                            x_test=payload.x_test,
                            y_test=payload.y_test)
        # Ensure we start to read from the beginning.
        data.seek(0)

        return data
