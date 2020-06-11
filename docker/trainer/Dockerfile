FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4:latest

RUN apt-get update && apt-get install libsndfile1 -y

# Install python packages
COPY src/trainer/requirements.txt /opt/requirements.txt
RUN pip install -U pip && \
    pip install -r /opt/requirements.txt --trusted-host pypiserver --ignore-installed

# Install GCSFUSE
RUN /bin/bash -c ' \
  export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; \
  echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -; \
  apt-get update; \
  apt-get install gcsfuse;'

# Copy training package, add to path and create directory to mount data to
COPY src/trainer/trainer /root/trainer
ENV PYTHONPATH "${PYTHONPATH}:/root/"
RUN mkdir /root/data

ENTRYPOINT ["python", "-u", "-m", "trainer.task"]
