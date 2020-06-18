FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

WORKDIR /opt

RUN apt-get update && apt-get install wget libsndfile1 -y

# Instll gcloud SDK
ARG gcloud_sdk='https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-297.0.1-linux-x86_64.tar.gz'
RUN wget -qO- $gcloud_sdk | tar -xz && \
    ./google-cloud-sdk/install.sh
ENV PATH "/opt/google-cloud-sdk/bin:$PATH"

# Install python packages
COPY src/trainer/requirements.txt /opt/requirements.txt
RUN pip install -U pip && \
    pip install --quiet -Iv future==0.18.2 && \
    pip install -r /opt/requirements.txt --trusted-host pypiserver --no-deps --ignore-installed

# Copy training package, add to path and create directory to mount data to
COPY src/trainer/trainer /opt/trainer
ENV PYTHONPATH "${PYTHONPATH}:/opt/"
RUN mkdir /opt/data

ENTRYPOINT ["python", "-u", "-m", "trainer.task"]
