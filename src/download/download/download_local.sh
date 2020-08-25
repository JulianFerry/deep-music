#!/bin/sh
root_dir=$1
dataset=$2

data_url="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-$dataset.jsonwav.tar.gz"
tar_file="nsynth-$dataset.tar.gz"

( cd $root_dir/data/raw &&
  wget -O $tar_file $data_url &&    # download tar
  tar -xzf $tar_file &&             # untar
  rm $tar_file                      # remove tar
)