#!/usr/bin/env bash

# Get image name and container arguments from the metadata
IMAGE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/image_name -H "Metadata-Flavor: Google")
CONTAINER_ARGS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/container_args -H "Metadata-Flavor: Google")
# This is needed to use a private GCR image
sudo HOME=/home/root /usr/bin/docker-credential-gcr configure-docker
# Run docker container! The logs will go to stack driver 
sudo HOME=/home/root  docker run --log-driver=gcplogs ${IMAGE_NAME} ${CONTAINER_ARGS}

# Shutdown compute instance
# Get the zone
zoneMetadata=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor:Google")
IFS=$'/'
zoneMetadataSplit=($zoneMetadata)
ZONE="${zoneMetadataSplit[3]}"
# Run delete on the current instance from within a container with gcloud installed 
docker run --entrypoint "gcloud" google/cloud-sdk:alpine compute instances delete ${HOSTNAME}  --delete-disks=all --zone=${ZONE}