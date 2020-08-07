#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir);
project_path=$(dirname $(dirname $package_path));

export GOOGLE_APPLICATION_CREDENTIALS=$project_path/credentials/deep-musik.json

# ( cd $project_path/data/raw &&
#     wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz -O temp.tar.gz &&
#     tar -xzf temp.tar.gz &&
#     rm temp.tar.gz
# )

```
import googleapiclient.discovery


def create_transfer_client():
    return googleapiclient.discovery.build('storagetransfer', 'v1')
```