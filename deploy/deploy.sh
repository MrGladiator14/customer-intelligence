#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ── Ensure az cli + extensions ──────────────────────────────────────────────
echo "==> Registering Azure Container App extensions and providers"
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
    ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv 2>/dev/null | tr -d '\r') && break
    echo "Waiting for ACR to become available... ($i/10)"
    sleep 3
done
[ -z "$ACR_SERVER" ] && { echo "ERROR: ACR not found after 30s"; exit 1; }
echo "ACR server: $ACR_SERVER"

# ── Build & Push Images using modular build script ─────────────────────────
echo "==> Executing image builds..."
source "${SCRIPT_DIR}/build.sh"

# ── Create Container Apps Environment ────────────────────────────────────────
echo "==> Creating Log Analytics workspace"
WORKSPACE_ID=$(az monitor log-analytics workspace create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${ENV_NAME}-logs" \
    --location "$LOCATION" \
    --query customerId -o tsv | tr -d '\r')

SHARED_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${ENV_NAME}-logs" \
    --query primarySharedKey -o tsv | tr -d '\r')

echo "==> Creating Container Apps Environment: $ENV_NAME"
az containerapp env create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ENV_NAME" \
    --location "$LOCATION" \
    --logs-workspace-id "$WORKSPACE_ID" \
    --logs-workspace-key "$SHARED_KEY" \
    -o table

# ── Resolve Container Apps Environment FQDN suffix ───────────────────────────
ACA_ENV_FQDN=$(az containerapp env show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ENV_NAME" \
    --query "properties.defaultDomain" -o tsv | tr -d '\r')

# ── Deploy FastAPI app ───────────────────────────────────────────────────────
echo "==> Deploying FastAPI app: $FASTAPI_APP"
az containerapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$FASTAPI_APP" \
    --environment "$ENV_NAME" \
    --image "${ACR_SERVER}/fastapi:${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" \
    --target-port 8000 \
    --ingress internal \
    --secrets \
        nvidia-api-key="${NVIDIA_API_KEY:?NVIDIA_API_KEY not set}" \
    --env-vars \
        NVIDIA_API_KEY=secretref:nvidia-api-key \
        MLFLOW_TRACKING_URI="http://${MLFLOW_APP}.internal.${ACA_ENV_FQDN}" \
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
    --ingress internal \
    --env-vars \
        MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db" \
        MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="*" \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 1 \
    --max-replicas 1 \
    -o table

# ── Deploy Nginx UI (ingress - external) ────────────────────────────────────
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
        FASTAPI_HOST="${FASTAPI_APP}.internal.${ACA_ENV_FQDN}" \
        MLFLOW_HOST="${MLFLOW_APP}.internal.${ACA_ENV_FQDN}" \
    --min-replicas 1 \
    --max-replicas 2 \
    -o table

# ── Output URLs ──────────────────────────────────────────────────────────────
UI_FQDN=$(az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NGINX_APP" \
    --query properties.configuration.ingress.fqdn \
    -o tsv | tr -d '\r')

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      Deployment Complete                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  UI + API Gateway : https://${UI_FQDN}              ║"
echo "║  MLflow UI        : https://${UI_FQDN}/mlflow/      ║"
echo "║  API (direct)     : https://${UI_FQDN}/api/health   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "All services are same-origin behind nginx - no CORS issues."
