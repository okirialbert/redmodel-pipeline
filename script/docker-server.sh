#!/bin/bash

set -u

if ! [ -x "$(command -v docker)" ]; then
    echo "ERROR: This script requires Docker"
    exit 1
fi

# The name of the image to run is specified by the constant below
DOCKER_IMAGE_NAME=tensorflow/serving

echo Pulling the TF Docker image: $DOCKER_IMAGE_NAME

docker pull $DOCKER_IMAGE_NAME

# Directories for model exported for serving
# TODO:: Change model directory

MODEL_DIR="$(pwd)/Dev/notebks/mh"
LOCAL_MODEL_DIR=savedmodel/redmodel
MODEL_NAME=redmodel
# Make sure the trained model is available
if ! [ -d "$LOCAL_MODEL_DIR" ]; then
  echo "ERROR: Model directory not found $LOCAL_MODEL_DIR"
  exit 1
fi

echo Starting the Model Server to serve from: $LOCAL_MODEL_DIR

# Container model dir
CONTAINER_MODEL_DIR=/models/redmodel

# Local port where to send inference requests
HOST_PORT=9000

# Where our container is listening
CONTAINER_PORT=8501

echo Model directory: $LOCAL_MODEL_DIR
  
docker run -p 8501:8501 \
  --mount type=bind,source=$LOCAL_MODEL_DIR,target=$CONTAINER_MODEL_DIR \
  -e MODEL_NAME=$MODEL_NAME -t $DOCKER_IMAGE_NAME
  

docker run -t --rm -p 8501:8501 \
    -v "$MODEL_DIR/$LOCAL_MODEL_DIR:$CONTAINER_MODEL_DIR" \
    -e MODEL_NAME=$MODEL_NAME \
    tensorflow/serving &


