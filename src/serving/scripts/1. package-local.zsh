#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));

export PYTHONPATH=$package_path
export GOOGLE_APPLICATION_CREDENTIALS='../../credentials/gs-access-key.json'

# No arguments to parse

# Test that the app works
( cd $package_path &&
  poetry run python3 -m $package_name.app \
)