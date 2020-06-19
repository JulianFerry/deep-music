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
run_config=$(cat $script_dir/run_list.json)
run_id=$(echo $run_config | jq ".[$run].id?")
fft_params=$(echo $run_config | jq ".[$run].fft_params?")
excluded_qualities=$(echo $run_config | jq ".[$run].excluded_qualities?")

echo "Using config from run_list.json for run $run_id:"
echo "--fft_params=$fft_params"
echo "--excluded_qualities=$excluded_qualities"
echo 

# # Run with parameters
( cd $package_path &&
  poetry run python3 -m $package_name.task \
    --dataset_path $project_path/data/raw/nsynth-train \
    --filters_path $project_path/data/interim/filters/train \
    --save_path $project_path/data/processed/spectrograms/config-$run_id \
    --fft_params $fft_params \
    --excluded_qualities $excluded_qualities
)