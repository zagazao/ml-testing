FROM python:3.8-buster

RUN adduser celeryworker && \
    mkdir -p /src/

# TODO: Split these calls for faster build time (src changes more often than req.)
ADD requirements.txt /tmp/

RUN pip3 install -r /tmp/requirements.txt

ADD . /tmp/cl/

RUN pip3 install redis

RUN cd /tmp/cl && pip3 install .

USER celeryworker

WORKDIR /src/

ENTRYPOINT ["celery", "-A", "mltest.celery_app", "worker", "--loglevel", "info"]
