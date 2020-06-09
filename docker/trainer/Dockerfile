FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4:latest

RUN apt-get update && apt-get install libsndfile1 -y

# Install python packages
COPY src/trainer /opt/trainer
RUN pip install -U pip && \
    pip install -r /opt/trainer/requirements.txt --trusted-host pypiserver --ignore-installed
RUN pip install /opt/trainer

RUN mkdir /root/data
ENTRYPOINT ["python", "-m", "trainer"]
