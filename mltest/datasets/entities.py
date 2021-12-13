from typing import List, Dict, Optional

import numpy as np
from bson import ObjectId
from pydantic import BaseModel, Field

# from carelabel.entities.tasks import Task
from enum import Enum


class Task(str, Enum):
    classification = 'classification'
    regression = 'regression'


class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")

        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class DatasetMetadata(BaseModel):
    name: str
    task: Task

    shape: List[int]
    n_samples: int
    full_dim: int
    meta_data: Dict
    # storage_path: str

    is_splitted: bool
    random_seed: Optional[int]
    split_idx: Optional[int]


class DatasetMetadataInDB(DatasetMetadata):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class DatasetPayload(BaseModel):
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class Dataset(DatasetPayload, DatasetMetadata):
    pass


class DatasetInDB(DatasetPayload, DatasetMetadataInDB):
    pass
