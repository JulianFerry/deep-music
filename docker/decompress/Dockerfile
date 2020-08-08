FROM python:3.7-slim-buster

COPY . /opt/decompress/
WORKDIR /opt/decompress
RUN pip install -U pip && \
    pip install -r requirements.txt

CMD exec gunicorn --bind :8080 --workers 1 --threads 1 --timeout 0 decompress:app