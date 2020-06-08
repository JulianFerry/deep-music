#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir)
project_path=$(dirname $(dirname $script_dir));
project_name=$(basename $project_path);
image_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$image_name
IMAGE_TAG=latest
image_name=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Run 
docker run \
  --volume $project_path/data/processed/time_intervals=1/resolution=5/:/root/data/ \
  $image_name \
    --job_dir /root/train-output/ \
    --data_dir /root/data/ \
    --epochs 1

docker logs -f $image_name