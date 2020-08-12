#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
export PYTHONPATH=$package_path

# Parse JSON config
config_list=$(cat $script_dir/configs.json)
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
    *)
        shift
        ;;
    esac
done

if [ $config_id = -1 ]; then
    echo -n "Enter preprocessing config ID (0 to $last_id): "
    read config_id; echo
fi

# If argument is numeric and the config ID exists
if [[ $config_id =~ ^[0-9]+$ ]] && [ $config_id -le $last_id ] && [ $config_id -ge 0 ]; then
    echo "Using preprocessing config id $config_id:"
    config=$(echo $config_list | jq ".[$config_id].config?")
    echo "${config} \n"
else
    echo "Error: No config found with id $config_id." && return 1
fi

# Run with config parameters
( cd $package_path &&
  source .venv/bin/activate &&
  python3 ${package_name}_main.py \
    --data_dir $project_path/data/raw/nsynth-test \
    --filters_dir $project_path/data/interim/filters/nsynth-test \
    --job_dir $project_path/data/processed/spectrograms/config-$config_id/nsynth-test \
    --config $config \
    --instruments '["keyboard_acoustic", "guitar_acoustic"]'
)