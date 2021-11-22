import os

import minio
# from pymongo import MongoClient
from pymongo import MongoClient


def minio_client_from_env():
    minio_url = os.getenv('MINIO_ENDPOINT')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY')
    # minio_secure = os.getenv('MINIO_SECURE') is not None
    minio_secure = False

    minio_client = minio.Minio(endpoint=minio_url,
                               access_key=minio_access_key,
                               secret_key=minio_secret_key,
                               secure=False)
    return minio_client


def mongo_client_from_env():
    mongo_host = os.getenv('MONGO_DB_HOST', 'localhost')
    mongo_port = int(os.getenv('MONGO_DB_PORT', 27017))

    mongo_username = os.getenv('MONGO_DB_USERNAME')
    mongo_password = os.getenv('MONGO_DB_PASSWORD')

    auth_args = {}
    if mongo_username is not None and mongo_password is not None:
        auth_args = {
            'username': mongo_username,
            'password': mongo_password
        }

    client = MongoClient(host=mongo_host,
                         port=mongo_port,
                         **auth_args)
    return client


mongo_client = mongo_client_from_env()
minio_client = minio_client_from_env()
