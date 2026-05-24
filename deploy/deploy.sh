#!/usr/bin/env bash
set -eo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
RESOURCE_GROUP="meridian-customer-intelligence"
LOCATION="${AZURE_LOCATION:-centralindia}"
ACR_NAME="${ACR_NAME:-meridianciacr}"
ENV_NAME="${ACA_ENV:-meridian-ci-env}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

FASTAPI_APP="fastapi-app"
MLFLOW_APP="mlflow-ui"
NGINX_APP="nginx-ui"

# ── Ensure az cli + extensions ──────────────────────────────────────────────
az extension add --name containerapp --upgrade --yes -o none
az provider register --namespace Microsoft.App --wait

echo "==> Creating resource group: $RESOURCE_GROUP"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o table

echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    -o table

for i in $(seq 1 10); do
    ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv 2>/dev/null) && break
    echo "Waiting for ACR to become available... ($i/10)"
    sleep 3
done
[ -z "$ACR_SERVER" ] && { echo "ERROR: ACR not found after 30s"; exit 1; }
echo "ACR server: $ACR_SERVER"

# ── Build & push images ──────────────────────────────────────────────────────
echo "==> Building and pushing images"
az acr build \
    --registry "$ACR_NAME" \
    --image "fastapi:${IMAGE_TAG}" \
    --file Dockerfile \
    .

az acr build \
    --registry "$ACR_NAME" \
    --image "mlflow:${IMAGE_TAG}" \
    --file docker/Dockerfile.mlflow \
    .

az acr build \
    --registry "$ACR_NAME" \
    --image "nginx-ui:${IMAGE_TAG}" \
    --file docker/Dockerfile.ui \
    .

# ── Create Container Apps Environment ────────────────────────────────────────
echo "==> Creating Log Analytics workspace"
WORKSPACE_ID=$(az monitor log-analytics workspace create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${ENV_NAME}-logs" \
    --location "$LOCATION" \
    --query id -o tsv)

SHARED_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${ENV_NAME}-logs" \
    --query primarySharedKey -o tsv)

echo "==> Creating Container Apps Environment: $ENV_NAME"
az containerapp env create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ENV_NAME" \
    --location "$LOCATION" \
    --logs-workspace-id "$WORKSPACE_ID" \
    --logs-workspace-key "$SHARED_KEY" \
    -o table

# ── Deploy FastAPI app ───────────────────────────────────────────────────────
echo "==> Deploying FastAPI app: $FASTAPI_APP"
az containerapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$FASTAPI_APP" \
    --environment "$ENV_NAME" \
    --image "${ACR_SERVER}/fastapi:${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" \
    --target-port 8000 \
    --ingress external false \
    --secrets \
        nvidia-api-key="${NVIDIA_API_KEY:?NVIDIA_API_KEY not set}" \
    --env-vars \
        NVIDIA_API_KEY=secretref:nvidia-api-key \
        MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db" \
        MLFLOW_EXPERIMENT_NAME="meridian-bank-marketing" \
        MLFLOW_INFERENCE_EXPERIMENT_NAME="meridian-bank-inference" \
    --min-replicas 1 \
    --max-replicas 3 \
    -o table

# ── Deploy MLflow UI ─────────────────────────────────────────────────────────
echo "==> Deploying MLflow UI: $MLFLOW_APP"
az containerapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$MLFLOW_APP" \
    --environment "$ENV_NAME" \
    --image "${ACR_SERVER}/mlflow:${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" \
    --target-port 5000 \
    --ingress external false \
    --env-vars \
        MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db" \
    --min-replicas 1 \
    --max-replicas 1 \
    -o table

# ── Deploy Nginx UI (ingress — external) ────────────────────────────────────
echo "==> Deploying Nginx UI: $NGINX_APP"
az containerapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NGINX_APP" \
    --environment "$ENV_NAME" \
    --image "${ACR_SERVER}/nginx-ui:${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" \
    --target-port 80 \
    --ingress external \
    --env-vars \
        FASTAPI_HOST="${FASTAPI_APP}:8000" \
        MLFLOW_HOST="${MLFLOW_APP}:5000" \
    --min-replicas 1 \
    --max-replicas 2 \
    -o table

# ── Output URLs ──────────────────────────────────────────────────────────────
UI_FQDN=$(az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NGINX_APP" \
    --query properties.configuration.ingress.fqdn \
    -o tsv)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      Deployment Complete                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  UI + API Gateway : https://${UI_FQDN}              ║"
echo "║  MLflow UI        : https://${UI_FQDN}/mlflow/      ║"
echo "║  API (direct)     : https://${UI_FQDN}/api/health   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "All services are same-origin behind nginx — no CORS issues."
