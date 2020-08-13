#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
project_name=$(basename $project_path);
service_name=$project_name-$package_name;

# GCP container registry container naming
project_id=$(gcloud config list project --format "value(core.project)")
image_tag=latest
image_name=eu.gcr.io/$project_id/$service_name:$image_tag


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
echo "Using training config:"
train_config='{
  "data_config_id": 0,
  "instruments": ["brass_electronic", "string_electronic"]
}'
echo $train_config
data_config_id=$(echo $train_config | jq ".data_config_id?")
echo


# User defined data paths
DATA_PATH="data/processed/spectrograms/config-$data_config_id/nsynth-train"
OUTPUT_PATH="trainer-output/docker_local"

# Test that the image works, using local data as a mounted volume
docker run --rm \
  --volume $project_path/data/:/opt/data \
  --volume $project_path/trainer-output/:/opt/trainer-output \
  --name $service_name \
  $image_name \
    --data_dir /opt/$DATA_PATH \
    --job_dir /opt/$OUTPUT_PATH \
    --train_config $train_config \
    --epochs 2