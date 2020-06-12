#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir)
project_path=$(dirname $(dirname $script_dir));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Generate requirements.txt for package
echo "Generating requirements.txt for $package_name"
( cd $project_path/src/$package_name && \
  poetry export --without-hashes -f requirements.txt > requirements.txt );
pypiserver_url="--extra-index-url http:\/\/pypiserver:8080"
if head -1 src/$package_name/requirements.txt | grep -q Warning; then
    >&2 head -1 src/$package_name/requirements.txt;
    return 1
else
    sed -i "1 s/.*/$pypiserver_url/" src/$package_name/requirements.txt
fi
sed -i "/torch==1.4.0/d" src/$package_name/requirements.txt
sed -i "/torchvision==0.5.0/d" src/$package_name/requirements.txt

# Stop and remove project container if it exists. Remove image if it exists
echo "Removing container $container_name and image $image_name"
docker ps       | grep -q $container_name && docker stop $container_name;
docker ps -a    | grep -q $container_name && docker rm $container_name;
docker image ls | grep -q $image_name && docker rmi -f $image_name;

# Build project image - needs the whole project as build context
echo "Building image $image_name"
( cd $project_path && \
  docker build \
    -t $image_name \
    -f $script_dir/Dockerfile \
    --network pypinet \
    $project_path )
