#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
project_name=$(basename $project_path);
service_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$service_name
IMAGE_TAG=latest
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

datasets=()
# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -r|--rebuild)
        # Rebuild image locally
        ( cd $project_path && . docker/$package_name/docker-build.zsh ) || return 1
        shift
        ;;
    -p|--push)
        # Push image to GCR
        ( docker push $image_name ) || return 1
        shift
        ;;
    train|valid|test)
        # Add train/valid/test to datasets
        datasets+=($1)
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Deploy image container to a compute engine VM
( gcloud compute instances create $package_name \
    --image-family=cos-stable \
    --image-project=cos-cloud  \
    --boot-disk-size 60G \
    --scopes "compute-rw,logging-write,storage-rw" \
    --metadata "image_name=$image_name,container_args=$datasets" \
    --metadata-from-file "startup-script=$script_dir/startup-script.sh" )


gcloud compute ssh $package_name    # This may fail because the instance creation is not done yet