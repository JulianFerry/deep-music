#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir);
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Stop and remove project container if it exists. Remove image if it exists
echo "Removing container $container_name and image $image_name"
docker ps       | grep -q $container_name && docker stop $container_name;
docker ps -a    | grep -q $container_name && docker rm $container_name;
docker image ls | grep -q $image_name && docker rmi -f $image_name;

# Build project image - needs the whole project as build context
echo "Building image $image_name"
( cd $package_path && \
  docker build \
    -t $image_name \
    -f $script_dir/Dockerfile \
    $package_path )
