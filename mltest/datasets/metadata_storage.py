import json
import uuid

import pymongo
from bson import ObjectId

from mltest.clients import mongo_client_from_env
from mltest.datasets.entities import DatasetMetadataInDB, DatasetMetadata, Task


class DatasetMetadataStorage(object):

    def load_meta_data_by_name(self, name) -> DatasetMetadataInDB:
        raise NotImplementedError

    def load_meta_data_by_id(self, _id) -> DatasetMetadataInDB:
        raise NotImplementedError

    def store_meta_data(self, meta_data: DatasetMetadata) -> DatasetMetadataInDB:
        raise NotImplementedError


class FileDatasetMetadataStorage(DatasetMetadataStorage):

    # TODO: Ensure locking
    # TODO: Optimize by avoid reloading (e.g. in memory mirror)
    def __init__(self, db_file='/tmp/metadata_db.json'):
        self.db_file = db_file

    def contains(self, meta_data: DatasetMetadata):
        with open(self.db_file, 'r') as fd:
            database = json.load(fd)
        for entry in database:
            if entry['name'] == meta_data.name:
                return True
        return False

    def load_meta_data_by_name(self, name) -> DatasetMetadataInDB:
        pass

    def load_meta_data_by_id(self, _id) -> DatasetMetadataInDB:
        pass

    def store_meta_data(self, meta_data: DatasetMetadata) -> DatasetMetadataInDB:
        with open(self.db_file, 'r') as fd:
            database = json.load(fd)
        if self.contains(meta_data):
            raise RuntimeError('Already exists.')


class MongoDatasetMetadataStorage(DatasetMetadataStorage):

    def __init__(self,
                 mongo_client: pymongo.MongoClient) -> None:
        super().__init__()
        self.mongo_client = mongo_client
        self.mongo_database = self.mongo_client['datasets']
        self.mongo_collection = self.mongo_database['metadata']

    def contains(self, meta_data: DatasetMetadata):

        query_result = self.mongo_collection.find_one({'name': meta_data.name})
        return query_result

    def store_meta_data(self, meta_data: DatasetMetadata) -> DatasetMetadataInDB:
        if self.contains(meta_data):
            raise RuntimeError('Entry already in db.')

        with_generated_id = DatasetMetadataInDB(**meta_data.dict())

        insert_result = self.mongo_collection.insert_one(with_generated_id.dict(by_alias=True))

        if insert_result.inserted_id is None:
            raise RuntimeError('Error during insertion.')
        return with_generated_id

    def _process_query_result(self, query_result: dict) -> DatasetMetadataInDB:
        # Validate item exists.
        if query_result is None:
            raise RuntimeError('NotFound')

        return DatasetMetadataInDB(**query_result)

    def load_meta_data_by_name(self, name) -> DatasetMetadataInDB:
        query_result = self.mongo_collection.find_one({'name': name})
        return self._process_query_result(query_result)

    def load_meta_data_by_id(self, _id) -> DatasetMetadataInDB:
        query_result = self.mongo_collection.find_one({'_id': ObjectId(_id)})
        return self._process_query_result(query_result)


if __name__ == '__main__':
    mongo_client = mongo_client_from_env()
    dummy_meta_data = DatasetMetadata(
        name=f'mnsit_{uuid.uuid4()}',
        task=Task.classification,
        shape=[28, 28, 1],
        n_samples=1000,
        full_dim=784,
        meta_data={},
        # storage_path=f'random_test_dataset_{uuid.uuid4()}',
        is_splitted=False
    )

    metadata_storage = MongoDatasetMetadataStorage(mongo_client)

    metadata_storage.store_meta_data(meta_data=dummy_meta_data)

    result = metadata_storage.load_meta_data_by_name(name=dummy_meta_data.name)

    print(dummy_meta_data)
