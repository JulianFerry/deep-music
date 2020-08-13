#!/bin/bash
abspath() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir);
project_path=$(abspath $(dirname $(dirname $package_path)));

dataset=$1  # User must pass argument 'train', 'valid' or 'test' when running this script
data_url="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-$dataset.jsonwav.tar.gz"
tar_file="nsynth-$dataset.tar.gz"

case $dataset in
    train|valid|test)
      ( cd $project_path/data/raw &&
        wget -O $tar_file $data_url &&                               # download tar
        tar -xzf $tar_file &&                                        # untar
        rm $tar_file                                                 # remove tar
      )
      ;;
    *)
      return 1
esac
