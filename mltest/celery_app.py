import io
import os

import cloudpickle
import redis
from celery import Celery, Task

from mltest.clients import minio_client

broker = os.getenv('CELERY_BROKER')

backend = os.getenv('CELERY_BACKEND', default='redis://localhost:6379/0')

print('BROKER:', broker)
print('BACKEND:', backend)

r = redis.Redis.from_url(backend)
print(r.client())
print(r.client().keys())


class TestRandomForestTask(Task):
    name = 'test_random_forest_task'

    def __init__(self):
        ...

    def get_model(self, task_id):
        obj = minio_client.get_object('jobs', f'{task_id}.model')

        model_bytes_io = io.BytesIO(obj.data)
        recovered_model = cloudpickle.load(model_bytes_io)
        return recovered_model

    def get_data(self, task_id):
        obj = minio_client.get_object('jobs', f'{task_id}.data')
        model_bytes_io = io.BytesIO(obj.data)
        # recovered_model = cloudpickle.load(model_bytes_io)
        ...

    def run(self, job_id):
        print('---')
        print('Job_id:', job_id)
        print('Test Random Forest Task')

        model = self.get_model(task_id=job_id)
        print(model)

        return {'abc': 1,
                'def': 2}


# Setup celery instance
celery_app = Celery(
    "tasks",
    backend=backend,
    broker=broker
)

celery_app.register_task(TestRandomForestTask)

# Configure queues.
# celery_app.conf.task_routes = {'fit_model_task': {'queue': 'initial_task'},
#                                'process_sub_task': {'queue': 'train_queue'}}
