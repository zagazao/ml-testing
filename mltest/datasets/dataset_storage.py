import uuid

import numpy as np

from mltest.clients import minio_client, mongo_client
from mltest.datasets.artifact_storage import DatasetArtifactStorage, S3MinioDatasetArtifactStorage
from mltest.datasets.encoder import NumpyDatasetEncoder, DatasetEncoder
from mltest.datasets.entities import DatasetMetadataInDB, Dataset, DatasetMetadata, DatasetPayload, Task
from mltest.datasets.metadata_storage import DatasetMetadataStorage, MongoDatasetMetadataStorage


class DatasetStorage(object):

    def __init__(self,
                 metadata_storage: DatasetMetadataStorage,
                 artifact_storage: DatasetArtifactStorage,
                 encoder: DatasetEncoder = NumpyDatasetEncoder()):
        self.metadata_storage = metadata_storage
        self.artifact_storage = artifact_storage

        self.encoder = encoder

    def _load_for_metadata(self, metadata: DatasetMetadataInDB):
        # Load artifact from store.r
        payload = self.artifact_storage.load_artifact(str(metadata.id))
        # Decode artifact.
        dataset = self.encoder.decode(payload)

        return Dataset(**metadata.dict(),
                       **dataset.dict())

    def load_by_name(self, name):
        meta_data = self.metadata_storage.load_meta_data_by_name(name)
        return self._load_for_metadata(meta_data)

    def load_by_id(self, _id):
        meta_data = self.metadata_storage.load_meta_data_by_id(_id)
        return self._load_for_metadata(meta_data)

    def store_dataset(self, dataset: Dataset):
        metadata_in_db = self.metadata_storage.store_meta_data(DatasetMetadata(**dataset.dict()))

        encoded_payload = self.encoder.encode(DatasetPayload(**dataset.dict()))

        self.artifact_storage.store_artifact(str(metadata_in_db.id), encoded_payload.getvalue())


if __name__ == '__main__':
    dataset_name = f'mnsit_{uuid.uuid4()}'

    meta_data = DatasetMetadata(
        name=dataset_name,
        task=Task.classification,
        shape=[28, 28, 1],
        n_samples=1000,
        full_dim=784,
        meta_data={},
        is_splitted=False
    )
    payload = DatasetPayload(
        x_train=np.random.randn(10, 10),
        x_test=np.random.randn(10, 10),
        y_train=np.random.randn(10),
        y_test=np.random.randn(10),
    )
    dataset = Dataset(**meta_data.dict(),
                      **payload.dict())

    storage = DatasetStorage(
        metadata_storage=MongoDatasetMetadataStorage(
            mongo_client=mongo_client
        ),
        artifact_storage=S3MinioDatasetArtifactStorage(
            minio_client=minio_client
        ),
    )

    storage.store_dataset(dataset)

    loaded_dataset = storage.load_by_name(dataset_name)

    print(loaded_dataset)
