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


if [ ! -f $package_path/audiolib-0.1.0.tar.gz ]; then
    echo "Building audiolib-0.1.0 as a local dependency for Dataflow"
    ( cd $package_path && \
    git clone -b '0.1.0' --single-branch --depth 1 ssh://git@github.com/JulianFerry/audiolib.git && \
    cd audiolib && \
    poetry build -f sdist && \
    cd $package_path && \
    mv audiolib/dist/audiolib-0.1.0.tar.gz . && \
    rm -rf audiolib )
fi


# User defined variables
BUCKET_NAME="deep-musik-data"
export GOOGLE_APPLICATION_CREDENTIALS="$project_path/credentials/gs-access-key.json"

# Run with config parameters
( cd $package_path &&
  source .venv/bin/activate &&
  python3 ${package_name}_main.py \
    --data_dir gs://$BUCKET_NAME/data/raw/nsynth-$dataset \
    --filters_dir gs://$BUCKET_NAME/data/interim/filters/nsynth-$dataset \
    --job_dir gs://$BUCKET_NAME/data/processed/spectrograms/config-$config_id/nsynth-$dataset \
    --config $config \
    --instruments '["brass_electronic", "string_electronic"]' \
    --runner dataflow \
    --num_workers 1 \
    --noauth_local_webserver \
    --setup_file $package_path/setup.py \
    --extra_package $package_path/audiolib-0.1.0.tar.gz
)