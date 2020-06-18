#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir);
project_path=$(dirname $(dirname $script_dir));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Rebuild image if arg is set
for arg in $@
do
  case $arg in
    -r|--rebuild)
      ( cd $project_path && . docker/$package_name/docker-build.zsh ) || return 1;;
  esac
done

# Run with local data as a mounted volume
docker run --rm \
  --volume $project_path/data/:/root/data/ \
  --name $container_name \
  $image_name \
    --data_dir /opt/data/processed/time_intervals=1/resolution=5/ \
    --job_dir /opt/train-output/ \
    --instruments "[keyboard_acoustic, guitar_acoustic]" \
    --epochs 1