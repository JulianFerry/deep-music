#!/bin/bash
abspath() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir);
project_path=$(abspath $(dirname $(dirname $package_path)));

dataset=$1  # User must pass argument 'train', 'valid' or 'test' when running this script
bucket_name="deep-musik-data"

( cd $project_path/data/raw &&
  gsutil -m cp -r nsynth-$dataset gs://$bucket_name/data/raw/ )