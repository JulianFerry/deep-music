FROM google/cloud-sdk:304.0.0

WORKDIR /home/

# Copy code
COPY src/download/download_local.sh src/download/download_local.sh
COPY src/download/gsutil_copy.sh src/download/gsutil_copy.sh
RUN mkdir data && mkdir data/raw

COPY docker/download/entrypoint.sh entrypoint.sh