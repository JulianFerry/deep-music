FROM google/cloud-sdk:304.0.0

RUN apt-get update && apt-get install wget -y
WORKDIR /home/

# Copy code
COPY src/download/download src/download/download
RUN mkdir data && mkdir data/raw

COPY docker/download/entrypoint.sh entrypoint.sh

ENTRYPOINT ["sh", "entrypoint.sh"]