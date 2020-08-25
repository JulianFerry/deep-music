FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

WORKDIR /opt

RUN apt-get update && apt-get install wget libsndfile1 git -y

# Instll gcloud SDK
ARG gcloud_sdk='https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-297.0.1-linux-x86_64.tar.gz'
RUN wget -qO- $gcloud_sdk | tar -xz && \
    ./google-cloud-sdk/install.sh
ENV PATH "/opt/google-cloud-sdk/bin:$PATH"

# Install python packages
COPY src/preprocessing /opt/preprocessing
COPY src/trainer /opt/trainer
COPY src/serving/requirements.txt /opt/serving/requirements.txt
WORKDIR /opt/serving
RUN pip install -U pip && \
    pip install --quiet -Iv future==0.18.2 && \
    pip install -r requirements.txt --no-deps --ignore-installed
COPY src/serving /opt/serving
ENV PYTHONPATH "${PYTHONPATH}:/opt/serving"

WORKDIR /opt/

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "serving.app:app"]
