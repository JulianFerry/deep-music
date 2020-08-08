#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
project_name=$(basename $project_path);
service_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$service_name
IMAGE_TAG=latest
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

datasets=()
# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -r|--rebuild)
        # Rebuild image
        ( cd $project_path && . docker/$package_name/docker-build.zsh ) || return 1
        shift
        ;;
    -p|--push)
        # Push image
        ( docker push $image_name ) || return 1
        shift
        ;;
    -d|--deploy)
        # Deploy image container on cloud run
        ( gcloud run deploy \
            --image $image_name \
            --platform managed \
            --region europe-west1 \
            --memory 1G
        ) || return 1
        shift
        ;;
    train|valid|test)
        # Add train/valid/test to datasets
        datasets+=($1)
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Authenticate current gcloud user to the service
gcloud run services add-iam-policy-binding \
  --platform managed \
  --region europe-west1 \
  $service_name \
    --member user:$(gcloud config list account --format "value(core.account)") \
    --role 'roles/run.invoker'

# Get cloud-run app url for the service
app_url=$(gcloud run services list --platform managed | \
          grep -Po "(https://$service_name.*?)(?= )")

# User settings
BUCKET_NAME='deep-musik-data'
ZIP_PATH_BASE='download.magenta.tensorflow.org/datasets/nsynth/nsynth-'
SAVE_PATH='data/raw/'

# Untar files specified in $datatset
if [ -n $app_url ]; then
    for dataset in "${datasets[@]}"; do
        curl \
          -X POST \
          -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
          -H "Content-Type: application/json" \
          --data "{
              \"bucket_name\": \"${BUCKET_NAME}\",
              \"zip_path\": \"${ZIP_PATH_BASE}${dataset}.jsonwav.tar.gz\",
              \"save_path\": \"${SAVE_PATH}\"
            }" \
          $app_url
    done
fi