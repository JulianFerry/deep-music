#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir);
project_path=$(dirname $(dirname $script_dir));
package_path=$project_path/src/$package_name
export PYTHONPATH=$package_path

# Parse JSON config
config_list=$(cat $script_dir/configs.json)
last_id=$(echo $config_list | jq ".[-1].id?")
config_id=$last_id

# Parse arguments
while [[ $# -gt 0 ]]
do
    case $1 in
    -i|--id)
        # If argument is numeric and the config ID exists
        if [[ $2 =~ ^[0-9]+$ ]] && [ $2 -le $last_id ] && [ $2 -ge 0 ]; then
            config_id=$2
            shift 2
        else
            echo "Error: No config found with id $2." && return 1
        fi
        ;;
    *)
        shift
        ;;
    esac
done

echo "\nUsing preprocessing config id $config_id:"
config=$(echo $config_list | jq ".[$config_id].config?")
echo "${config} \n"

# Run with config parameters
( cd $package_path &&
  poetry run python3 -m $package_name.task \
    --data_dir $project_path/data/raw/nsynth-train \
    --filters_dir $project_path/data/interim/filters/train \
    --job_dir $project_path/data/processed/spectrograms/config-$config_id \
    --config $config \
    --instruments '["keyboard_acoustic", "guitar_acoustic"]'
)