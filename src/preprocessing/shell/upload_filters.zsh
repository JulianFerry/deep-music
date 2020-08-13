#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
export PYTHONPATH=$package_path

BUCKET_NAME="deep-musik-data"

export GOOGLE_APPLICATION_CREDENTIALS=$project_path/credentials/gs-access-key.json
gsutil -m cp -r $project_path/data/interim/filters gs://$BUCKET_NAME/data/interim/