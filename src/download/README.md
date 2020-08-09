# Option A: Local

## Local download
- Run `download_local.zsh DATASETS` with `DATASETS` equal to any combination of `train`, `valid` and `test` 

## Upload to cloud
- Run `gsutil_copy.zsh DATASETS` with `DATASETS` equal to any combination of `train`, `valid` and `test` 

# Option B: Cloud

## To run the above two steps in compute engine
- Run `3. gcloud-run.zsh DATASETS` with `DATASETS` equal to any combination of `train`, `valid` and `test`. Specify `-r` to rebuild the docker image and `-p` to push the image before running.