#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir);
project_path=$(dirname $(dirname $package_path));

dataset=$1  # User must pass argument 'train', 'valid' or 'test' when running this script
data_url="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-$dataset.jsonwav.tar.gz"
tar_file="nsynth-$dataset.tar.gz"

( cd $project_path/data/raw &&
    curl -f $data_url -o $tar_file &&                               # download tar
    bytes=$(stat -c %s $tar_file) &&                                # calculate bytes
    md5=$(openssl md5 -binary $tar_file | openssl enc -base64) &&   # calculate md5
    info="${data_url}\t${bytes}\t${md5}" &&                         # concatenate
    sed -i "s?${data_url}.*?${info}?g" urls.tsv &&                  # add to urls.tsv
    tar -xzf $tar_file &&                                           # untar
    rm $tar_file                                                    # remove tar
)

