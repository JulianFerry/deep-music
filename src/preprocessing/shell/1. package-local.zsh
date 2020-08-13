#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));

# Parse JSON config
config_list=$(cat $script_dir/data_configs.json)
last_id=$(echo $config_list | jq ".[-1].id?")
config_id=-1

# Parse arguments
dataset=""
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

# Read numeric config between 0 and last_id if not specified
while ! ([[ $config_id =~ ^[0-9]+$ ]] && [ $config_id -le $last_id ] && [ $config_id -ge 0 ]); do
    echo -n "Enter preprocessing config ID (0 to $last_id): "
    read config_id;
done

# Read dataset (train/valid/test) if not specified
while ! [[ $dataset =~ ^(train|valid|test)$ ]]; do
    echo -n "Enter dataset name (train/valid/test): "
    read dataset;
done
echo

echo "Using preprocessing config id $config_id for nsynth-$dataset:"
config=$(echo $config_list | jq ".[$config_id].config?")
echo "${config} \n"

# Run with config parameters
( cd $package_path &&
  source .venv/bin/activate &&
  python3 ${package_name}_main.py \
    --data_dir $project_path/data/raw/nsynth-$dataset \
    --filters_dir $project_path/data/interim/filters/nsynth-$dataset \
    --job_dir $project_path/data/processed/spectrograms/config-$config_id/nsynth-$dataset \
    --config $config \
    --instruments '["brass_electronic", "string_electronic"]'
)