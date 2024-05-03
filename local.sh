#!/bin/sh

IMAGE_NAME="eon-chatbot"
CONTAINER_NAME="eon-chatbot"
PORT_MAPPING="8501:8501"

# Check if the container exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME exists. Attempting to stop and remove..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    echo "Container $CONTAINER_NAME has been stopped and removed."
fi

echo "Building Docker image $IMAGE_NAME..."
docker build --no-cache -t $IMAGE_NAME .

echo "Running Docker container from image $IMAGE_NAME on port $PORT_MAPPING..."
docker run --name $CONTAINER_NAME -p $PORT_MAPPING $IMAGE_NAME
