#!/bin/sh
for dataset in "$@"
do
    echo "Downloading files for dataset: $dataset"
    sh src/download/download_local.sh $dataset
    echo "Copying files to cloud storage for dataset: $dataset"
    sh src/download/gsutil_copy.sh $dataset
done