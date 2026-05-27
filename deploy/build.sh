#!/usr/bin/env bash
set -eo pipefail

# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "==> Logging in to ACR: $ACR_NAME"
az acr login --name "$ACR_NAME"

echo "==> Building and pushing Docker images (tag: $IMAGE_TAG)"
cd "${SCRIPT_DIR}/.."

if [[ -z "$BUILD_TARGET" || "$BUILD_TARGET" == "all" || "$BUILD_TARGET" == "fastapi" ]]; then
    echo "==> [1/3] Building & pushing FastAPI app..."
    docker build -t "${ACR_SERVER}/fastapi:${IMAGE_TAG}" -f docker/Dockerfile.fastapi .
    docker push "${ACR_SERVER}/fastapi:${IMAGE_TAG}"
fi

if [[ -z "$BUILD_TARGET" || "$BUILD_TARGET" == "all" || "$BUILD_TARGET" == "mlflow" ]]; then
    echo "==> [2/3] Building & pushing MLflow server..."
    docker build -t "${ACR_SERVER}/mlflow:${IMAGE_TAG}" -f docker/Dockerfile.mlflow .
    docker push "${ACR_SERVER}/mlflow:${IMAGE_TAG}"
fi

if [[ -z "$BUILD_TARGET" || "$BUILD_TARGET" == "all" || "$BUILD_TARGET" == "nginx" ]]; then
    echo "==> [3/3] Building & pushing Nginx frontend..."
    docker build -t "${ACR_SERVER}/nginx-ui:${IMAGE_TAG}" -f docker/Dockerfile.ui .
    docker push "${ACR_SERVER}/nginx-ui:${IMAGE_TAG}"
fi

echo "==> All images built and pushed successfully!"
