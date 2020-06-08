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

# Generate requirements.txt for package
echo "Generating requirements.txt for $package_name"
( cd src/trainer && \
  poetry export --without-hashes -f requirements.txt > requirements.txt );
pypiserver_url="--extra-index-url http:\/\/pypiserver:8080"
gsed -i "1 s/.*/$pypiserver_url/" src/trainer/requirements.txt
gsed -i "/torch==1.4.0/d" src/trainer/requirements.txt
gsed -i "/torchvision==0.5.0/d" src/trainer/requirements.txt

# Stop and remove project container if it exists. Remove image if it exists
echo "Removing container $image_name and image $image_name"
docker ps       | grep -q $image_name && docker stop $image_name;
docker ps -a    | grep -q $image_name && docker rm $image_name;
docker image ls | grep -q $image_name && docker rmi -f $image_name;

# Build project image
echo "Building image $image_name"
if docker build \
  -t $image_name \
  -f $script_dir/Dockerfile \
  --network pypinet \
  $project_path; then \
    echo "Image built"
fi
