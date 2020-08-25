FROM google/cloud-sdk:304.0.0

RUN apt-get update && apt-get install wget -y
WORKDIR /home/

# Copy code
COPY download /home/download
RUN mkdir data && mkdir data/raw

COPY docker/entrypoint.sh entrypoint.sh

ENTRYPOINT ["sh", "entrypoint.sh"]