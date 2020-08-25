#!/bin/bash
root_dir=$1
dataset=$2
bucket_name=$3

( cd $root_dir/data/raw &&
  gsutil -m cp -r nsynth-$dataset gs://$bucket_name/data/raw/ )