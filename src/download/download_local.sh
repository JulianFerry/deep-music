#!/bin/bash
abspath() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
script_dir=$(dirname $0:A);
project_path=$(abspath $(dirname $(dirname $script_dir)));

dataset=$1  # User must pass argument 'train', 'valid' or 'test' when running this script
data_url="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-$dataset.jsonwav.tar.gz"
tar_file="nsynth-$dataset.tar.gz"

( cd $project_path/data/raw &&
    curl -f $data_url -o $tar_file &&                               # download tar
    tar -xzf $tar_file &&                                           # untar
    rm $tar_file                                                    # remove tar
)

