#!/bin/zsh
script_dir=$(dirname $0:A);
package_dir=$(dirname $script_dir);
project_path=$(dirname $(dirname $package_dir));
export PYTHONPATH=$project_path/src/trainer

# Test that the package works
poetry run python3 -m trainer \
    --job_dir $project_path/data/train-output/ \
    --data_dir $project_path/data/processed/time_intervals=1/resolution=5 \
    --epochs 1