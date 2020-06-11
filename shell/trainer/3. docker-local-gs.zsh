#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir);
project_path=$(dirname $(dirname $script_dir));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# gcloud AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest

export BUCKET_NAME=deep-musik-data
export IMAGE_URI=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Run with cloud storage credentials as a volume
docker run --rm \
  --cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined \
  --volume $project_path/credentials/:/root/credentials/:ro \
  --name $container_name \
  $IMAGE_URI \
    --data_dir gs://$BUCKET_NAME/data/processed/time_intervals=1/resolution=5/ \
    --job_dir /root/train-output/ \
    --epochs 1