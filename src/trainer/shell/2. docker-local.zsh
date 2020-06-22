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

# Mock config parsing
echo "Using training data config:"
data_config='{
  "data_id": 0,
  "instruments": ["brass_electronic", "string_electronic"]
}'
echo $data_config
data_id=$(echo $data_config | jq ".data_id?")
echo "Applied to data preprocessed with config $data_id"
echo


# Data paths
data_path=data/processed/spectrograms/config-$data_id/nsynth-train
output_path=output/${package_name}_local

# Test that the image works, using local data as a mounted volume
docker run --rm \
  --volume $project_path/data/:/opt/data \
  --volume $project_path/output/:/opt/output \
  --name $container_name \
  $image_name \
    --data_dir /opt/$data_path \
    --job_dir /opt/$output_path \
    --data_config $data_config \
    --epochs 1