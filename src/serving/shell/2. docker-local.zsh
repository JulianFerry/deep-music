#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG


# No JSON config file to parse

# Parse arguments
for arg in $@
do
    case $arg in
    -r|--rebuild)
        # Rebuild image
        ( cd $project_path && . docker/$package_name/docker-build.zsh ) || return 1
        ;;
    esac
done


# Test that the image works, using mounted credentials
docker run --rm \
  --volume $project_path/credentials/:/opt/credentials/:ro \
  --env GOOGLE_APPLICATION_CREDENTIALS='/opt/credentials/gs-access-key.json' \
  --name $container_name \
  --publish 8080:8080 \
  $image_name