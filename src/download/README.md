## Local download
- Run `download_local.zsh $DATASET` with `$DATASET` equal to `train`, `valid`, or `test` 

## Cloud transfer:
- Create  project on Google Cloud
- Create a new storage bucket
- Create a transfer job on the [Cloud Transfer Service](https://console.cloud.google.com/transfer/cloud/) console, with "List of object URLs" as the source and paste the following URL: https://raw.githubusercontent.com/JulianFerry/deep-music/dataflow/data/raw/urls.tsv