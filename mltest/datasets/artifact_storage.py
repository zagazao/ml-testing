import io
import os.path
import uuid

import minio

from mltest.clients import minio_client_from_env


class DatasetArtifactStorage(object):

    # TODO: Should we use bytes or bytesIo as byte type? Atm we use bytes
    def load_artifact(self,
                      path: str) -> bytes:
        raise NotImplementedError

    def store_artifact(self,
                       path: str,
                       payload: bytes):
        raise NotImplementedError


class FileDatasetArtifactStorage(DatasetArtifactStorage):

    def __init__(self,
                 storage_directory='/tmp/file_storage/') -> None:
        super().__init__()
        self.storage_directory = storage_directory

        if not os.path.exists(storage_directory):
            os.makedirs(storage_directory)

    def __get_storage_location(self,
                               path) -> str:
        return os.path.join(self.storage_directory, path)

    def load_artifact(self,
                      path) -> bytes:
        storage_location = self.__get_storage_location(path)
        if not os.path.exists(storage_location):
            raise FileNotFoundError('File does not exist.')

        with open(storage_location, 'rb') as fd:
            payload = fd.read()
            print(type(payload))
        return payload

    def store_artifact(self,
                       path: str,
                       payload: bytes):
        storage_location = self.__get_storage_location(path)

        if os.path.exists(storage_location):
            raise FileNotFoundError('File already exists.')

        with open(storage_location, 'wb') as fd:
            fd.write(payload)


class S3MinioDatasetArtifactStorage(DatasetArtifactStorage):

    def __init__(self,
                 minio_client: minio.Minio):
        self.minio_client = minio_client

    def load_artifact(self, path: str) -> bytes:
        s3_response = self.minio_client.get_object('datasets', path)
        if s3_response.status != 200:
            raise Exception('Invalid http code.')

        return s3_response.data

    def store_artifact(self, path: str, payload: bytes):
        # TODO: Check len(payload)
        self.minio_client.put_object('datasets', path, io.BytesIO(payload), len(payload))


if __name__ == '__main__':

    my_str = '1414bsfsfs'
    payload = io.BytesIO(bytes(my_str, encoding='utf8'))

    name = f'dataset_{uuid.uuid4()}'

    artifcat_storage = FileDatasetArtifactStorage()

    artifcat_storage.store_artifact(name, payload.getvalue())
    reconstructed = artifcat_storage.load_artifact(name)

    minio_client = minio_client_from_env()

    s3_store = S3MinioDatasetArtifactStorage(minio_client)

    s3_store.store_artifact(name, payload.getvalue())
    s3_reconstructed = s3_store.load_artifact(name)

    print(reconstructed)
