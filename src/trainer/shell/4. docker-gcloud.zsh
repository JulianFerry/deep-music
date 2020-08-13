#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
project_name=$(basename $project_path);
service_name=$project_name-$package_name;

# GCP container registry naming
project_id=$(gcloud config list project --format "value(core.project)")
image_tag=latest
image_uri=eu.gcr.io/$project_id/$service_name:$image_tag


# Parse JSON config file
config_list=$(cat $script_dir/train_configs.json)
last_id=$(echo $config_list | jq ".[-1].id?")
config_id=-1

# Parse arguments
while [[ $# -gt 0 ]]
do
    case $1 in
    -i|--id)
        config_id=$2
        shift 2
        ;;
    -r|--rebuild)
        # Rebuild image
        ( cd $project_path && . docker/$package_name/docker-build.zsh ) || return 1
        shift
        ;;
    -p|--push)
        # Push image
        docker push $image_uri
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Read numeric config between 0 and last_id if not specified
while ! ([[ $config_id =~ ^[0-9]+$ ]] && [ $config_id -le $last_id ] && [ $config_id -ge 0 ]); do
    echo -n "Enter training config ID (0 to $last_id): "
    read config_id;
done

# Parse config
echo "Using training config id $config_id:"
train_config=$(echo $config_list | jq ".[$config_id].config?")
echo $train_config
data_config_id=$(echo $config_list | jq ".[$config_id].config.data_config_id?")
echo


# Data paths
BUCKET_NAME="deep-musik-data"
REGION='europe-west1'
DATA_PATH="data/processed/spectrograms/config-$data_config_id/nsynth-train"
JOB_DIR="config${config_id}/$(date +%y%m%d_%H%M%S)"
job_name=${JOB_DIR//\//_};  # replace / with _ for job name
OUTPUT_PATH="trainer-output/${JOB_DIR}"

# Submit training job to gcloud AI platform
gcloud ai-platform jobs submit training $job_name \
  --region $REGION \
  --master-image-uri $image_uri \
  -- \
    --data_dir=gs://$BUCKET_NAME/$DATA_PATH \
    --job_dir=gs://$BUCKET_NAME/$OUTPUT_PATH \
    --train_config $train_config \
    --epochs=100