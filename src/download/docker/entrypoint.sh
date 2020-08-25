#!/bin/sh
root_dir=$1
bucket_name=$2
shift; shift;

for dataset in "$@"
do
    case $dataset in
    train|valid|test)
        echo "Downloading files for dataset: $dataset"
        sh download/download_local.sh $root_dir $dataset
        echo "Copying files to cloud storage for dataset: $dataset"
        sh download/gsutil_copy.sh $root_dir $dataset $bucket_name
        ;;
    esac
done