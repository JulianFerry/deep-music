FROM python:3.7-slim-buster

COPY src/decompress /opt/decompress/
WORKDIR /opt/decompress
RUN pip install -U pip && \
    pip install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/opt/decompress"

WORKDIR /home/
COPY credentials credentials
COPY docker/decompress/entrypoint.sh .

ENTRYPOINT /bin/bash