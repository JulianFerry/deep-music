FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4:latest

RUN apt-get update && apt-get install libsndfile1 -y

# Install python packages
COPY src/trainer/requirements.txt /opt/requirements.txt
RUN pip install -U pip && \
    pip install -r /opt/requirements.txt --trusted-host pypiserver --ignore-installed

# Copy training package, add to path and create directory to mount data to
COPY src/trainer/trainer /root/trainer
ENV PYTHONPATH "${PYTHONPATH}:/root/"
RUN mkdir /root/data

ENTRYPOINT ["python", "-u", "-m", "trainer.task"]
