export BUCKET_NAME=custom_containers
export MODEL_DIR=example_model_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR \
  --epochs=10