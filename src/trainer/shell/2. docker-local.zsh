#!/bin/zsh
script_dir=$(dirname $0:A);
package_dir=$(dirname $script_dir);
package_name=$(basename $package_dir);
project_path=$(dirname $(dirname $package_dir));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest
image_name=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Run 
docker run --rm \
  --volume $project_path/data/processed/time_intervals=1/resolution=5/:/root/data/ \
  --name $container_name \
  $image_name \
    --job_dir /root/train-output/ \
    --data_dir /root/data/ \
    --epochs 1

docker logs -f $container_name