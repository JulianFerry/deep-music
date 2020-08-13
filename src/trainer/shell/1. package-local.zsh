#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
export PYTHONPATH=$package_path


# No JSON config file to parse

# No arguments to parse

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
OUTPUT_PATH="trainer-output/local"

# Test that the package works
( cd $package_path &&
  poetry run python3 -m $package_name.task \
    --data_dir $project_path/$DATA_PATH \
    --job_dir $project_path/$OUTPUT_PATH \
    --train_config $train_config \
    --epochs 2
)