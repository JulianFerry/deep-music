#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir);
project_path=$(dirname $(dirname $script_dir));
package_path=$project_path/src/$package_name
export PYTHONPATH=$package_path

if [[ $1 =~ ^[0-9]+$ ]]; then
    run=$1;
else
    echo "Run number not specified"
    run=-1
fi

# Parse JSON config
run_config=$(cat $script_dir/configs.json)
run_id=$(echo $run_config | jq ".[$run].id?")
config=$(echo $run_config | jq ".[$run].config?")

echo "Using config from configs.json for run $run_id:"
echo "$config"
echo 

# # Run with parameters
( cd $package_path &&
  poetry run python3 -m $package_name.task \
    --dataset_path $project_path/data/raw/nsynth-train \
    --filters_path $project_path/data/interim/filters/train \
    --save_path $project_path/data/processed/spectrograms/config-$run_id \
    --config $config \
)