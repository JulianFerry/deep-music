#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
export PYTHONPATH=$package_path


# No JSON config file to parse

# No arguments to parse

# Mock config parsing
echo "Using training data config:"
data_config='{
  "data_id": 0,
  "instruments": ["brass_electronic", "string_electronic"]
}'
echo $data_config
data_id=$(echo $data_config | jq ".data_id?")
echo "Applied to data preprocessed with data_config $data_id"
echo


# Data paths
data_path=data/processed/spectrograms/config-$data_id/nsynth-train
output_path=output/${package_name}_local

# Test that the package works
( cd $package_path &&
  poetry run python3 -m $package_name.task \
    --data_dir $project_path/$data_path \
    --job_dir $project_path/$output_path \
    --data_config $data_config \
    --epochs 2
)