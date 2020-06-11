#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir);
project_path=$(dirname $(dirname $script_dir));
package_path=$project_path/src/$package_name
export PYTHONPATH=$package_path

# Test that the package works
( cd $package_path &&
  poetry run python3 -m trainer.task \
    --data_dir $project_path/data/processed/time_intervals=1/resolution=5 \
    --job_dir $project_path/train-output/ \
    --instruments "[brass_electronic, string_electronic]" \
    --epochs 1 )