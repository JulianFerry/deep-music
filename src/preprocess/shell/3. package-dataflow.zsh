#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
export PYTHONPATH=$package_path

BUCKET_NAME=deep-musik-data
export GOOGLE_APPLICATION_CREDENTIALS=$project_path/credentials/gs-access-key.json

# Parse JSON config
config_list=$(cat $script_dir/configs.json)
last_id=$(echo $config_list | jq ".[-1].id?")
config_id=-1
dataset=""

# Parse arguments
while [[ $# -gt 0 ]]
do
    case $1 in
    -i|--id)
        config_id=$2
        shift 2
        ;;
    -d|--dataset)
        dataset=$2
        shift 2
        ;;
    *)
        shift
        ;;
    esac
done

while ! ([[ $config_id =~ ^[0-9]+$ ]] && [ $config_id -le $last_id ] && [ $config_id -ge 0 ]); do
    echo -n "Enter preprocessing config ID (0 to $last_id): "
    read config_id;
done

while ! [[ "$dataset" =~ ^(train|valid|test)$ ]]; do
    echo -n "Enter dataset name (train/valid/test): "
    read dataset;
done
echo

# If argument is numeric and the config ID exists
echo "Using preprocessing config id $config_id for nsynth-$dataset:"
config=$(echo $config_list | jq ".[$config_id].config?")
echo "${config} \n"

# Run with config parameters
( cd $package_path &&
  source .venv/bin/activate &&
  python3 ${package_name}_main.py \
    --data_dir gs://$BUCKET_NAME/data/raw/nsynth-$dataset \
    --filters_dir gs://$BUCKET_NAME/data/interim/filters/nsynth-$dataset \
    --job_dir gs://$BUCKET_NAME/data/processed/spectrograms/config-$config_id/nsynth-$dataset \
    --config $config \
    --instruments '["keyboard_acoustic", "guitar_acoustic"]' \
    --runner dataflow \
    --noauth_local_webserver \
    --setup_file $package_path/setup.py \
    --extra_package $project_path/../audiolib/dist/audiolib-0.1.0.tar.gz
)